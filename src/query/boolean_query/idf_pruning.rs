use crate::query::term_query::TermScorer;
use crate::{DocId, DocSet, Score, TERMINATED};

/// Statistics collected during IDF pruning execution.
#[derive(Debug, Default)]
pub struct IdfPruningStats {
    pub num_seeks: u64,
    pub num_advances: u64,
    pub candidates_evaluated: u64,
    pub candidates_emitted: u64,
    pub phase1_candidates: u64,
    pub phase2_candidates: u64,
    pub num_terms: u32,
    pub num_absolutely_essential_at_end: u32,
    pub initial_min_essential_matches: u32,
    pub initial_num_essential: u32,
    pub initial_threshold: Score,
    pub final_threshold: Score,
}

/// IDF-only scoring with intersection-based pruning.
///
/// When term frequencies are not stored (`IndexRecordOption::Basic`), each term
/// contributes a constant IDF score to every document that contains it. This makes
/// `max_score == actual_score`, enabling perfectly tight bounds and a powerful
/// optimization: terms whose IDF exceeds `total_idf - threshold` are "absolutely
/// essential" — every top-K result MUST contain them. Their posting lists can be
/// intersected to dramatically reduce candidates.
///
/// The algorithm has three phases:
/// - Phase 0: Bootstrap threshold by sampling documents from the rarest term
/// - Phase 1: MaxScore iteration (union of essential terms) until absolutely essential terms exist
/// - Phase 2: Intersection of absolutely essential terms + lazy scoring of remainder
pub fn idf_pruning(
    mut scorers: Vec<TermScorer>,
    mut threshold: Score,
    top_k: usize,
    callback: &mut dyn FnMut(u32, Score) -> Score,
) -> IdfPruningStats {
    let mut stats = IdfPruningStats {
        num_terms: scorers.len() as u32,
        ..Default::default()
    };

    if scorers.is_empty() {
        return stats;
    }

    let idfs: Vec<Score> = scorers.iter().map(|s| s.idf_score()).collect();
    let total_idf: Score = idfs.iter().sum();

    if total_idf <= threshold {
        stats.initial_threshold = threshold;
        stats.final_threshold = threshold;
        return stats;
    }

    // Partition into essential / non-essential based on MaxScore prefix-sum
    let mut all_indices: Vec<usize> = (0..scorers.len()).collect();
    all_indices.sort_by(|&a, &b| {
        idfs[a]
            .partial_cmp(&idfs[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Phase 0: Analytical threshold bootstrap from global doc frequencies.
    //
    // If a single term has doc_freq >= K, then at least K documents score >= that
    // term's IDF. We walk from highest IDF (rarest) to find the best such term.
    //
    // We only consider individual terms (not accumulated counts) because documents
    // can appear in multiple terms' posting lists, making accumulation unsound.
    if top_k > 0 && threshold <= Score::MIN + 1.0 {
        for &idx in all_indices.iter().rev() {
            if scorers[idx].global_doc_freq() >= top_k as u64 {
                // Subtract a margin to avoid floating-point issues where
                // `total_idf - threshold` rounds below a term's IDF, falsely
                // triggering absolutely-essential partition.
                let margin = total_idf * 1e-6;
                let bootstrap = idfs[idx] - margin;
                if bootstrap > threshold {
                    threshold = bootstrap;
                }
                break;
            }
        }
    }

    stats.initial_threshold = threshold;

    // Re-check after Phase 0 may have raised threshold
    if total_idf <= threshold {
        stats.final_threshold = threshold;
        return stats;
    }

    let split = prefix_sum_split(&all_indices, &idfs, threshold);
    let mut non_essential: Vec<usize> = all_indices[..split].to_vec();
    let mut essential: Vec<usize> = all_indices[split..].to_vec();

    // Defensive: if f32 rounding made essential empty, put everything essential
    if essential.is_empty() {
        essential = all_indices.clone();
        non_essential.clear();
    }

    let mut non_essential_idf_sum: Score = non_essential.iter().map(|&i| idfs[i]).sum();

    // Check if we can go straight to Phase 2
    let (abs, rest) = partition_absolutely_essential(&essential, &idfs, total_idf, threshold);
    if !abs.is_empty() {
        let mut remaining = rest;
        remaining.extend_from_slice(&non_essential);
        phase2(
            &mut scorers,
            &idfs,
            total_idf,
            abs,
            remaining,
            &mut threshold,
            callback,
            &mut stats,
        );
    } else {
        phase1(
            &mut scorers,
            &idfs,
            total_idf,
            &mut essential,
            &mut non_essential,
            &mut non_essential_idf_sum,
            &mut threshold,
            callback,
            &mut stats,
        );
    }

    stats.final_threshold = threshold;
    stats
}

/// Returns the index where the prefix sum of IDFs first exceeds threshold.
/// indices_by_idf_asc must be sorted by IDF ascending.
/// Elements before the split are non-essential, elements from split onward are essential.
fn prefix_sum_split(indices_by_idf_asc: &[usize], idfs: &[Score], threshold: Score) -> usize {
    let mut sum = 0.0f32;
    for (i, &idx) in indices_by_idf_asc.iter().enumerate() {
        sum += idfs[idx];
        if sum > threshold {
            return i;
        }
    }
    indices_by_idf_asc.len()
}

/// Partition terms into absolutely essential and rest.
/// A term is absolutely essential if IDF > total_idf - threshold.
fn partition_absolutely_essential(
    terms: &[usize],
    idfs: &[Score],
    total_idf: Score,
    threshold: Score,
) -> (Vec<usize>, Vec<usize>) {
    let cutoff = total_idf - threshold;
    let mut abs = Vec::new();
    let mut rest = Vec::new();
    for &idx in terms {
        if idfs[idx] > cutoff {
            abs.push(idx);
        } else {
            rest.push(idx);
        }
    }
    (abs, rest)
}

/// Compute the minimum number of essential term matches needed to exceed threshold.
///
/// A document matching m essential terms has max score = sum of m largest essential IDFs
/// + non_essential_idf_sum. Returns the smallest m where this exceeds threshold.
fn compute_min_essential_matches(
    essential: &[usize],
    idfs: &[Score],
    non_essential_idf_sum: Score,
    threshold: Score,
) -> usize {
    let mut sorted_idfs: Vec<Score> = essential.iter().map(|&i| idfs[i]).collect();
    sorted_idfs.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = non_essential_idf_sum;
    for (i, &idf) in sorted_idfs.iter().enumerate() {
        cumulative += idf;
        if cumulative > threshold {
            return i + 1;
        }
    }
    essential.len()
}

/// Phase 1: MaxScore iteration with union of essential terms.
fn phase1(
    scorers: &mut [TermScorer],
    idfs: &[Score],
    total_idf: Score,
    essential: &mut Vec<usize>,
    non_essential: &mut Vec<usize>,
    non_essential_idf_sum: &mut Score,
    threshold: &mut Score,
    callback: &mut dyn FnMut(u32, Score) -> Score,
    stats: &mut IdfPruningStats,
) {
    let mut min_essential_matches = compute_min_essential_matches(
        essential, idfs, *non_essential_idf_sum, *threshold,
    );
    stats.initial_min_essential_matches = min_essential_matches as u32;
    stats.initial_num_essential = essential.len() as u32;

    loop {
        if essential.is_empty() {
            break;
        }

        // Find pivot: kth smallest doc among essential scorers
        let k = min_essential_matches.min(essential.len());
        essential.select_nth_unstable_by_key(k - 1, |&i| scorers[i].doc());
        let pivot = scorers[essential[k - 1]].doc();

        if pivot == TERMINATED {
            break;
        }

        // Seek lagging scorers (doc < pivot) forward to pivot.
        // After select_nth, elements [0..k-1] have doc <= pivot, [k..] have doc >= pivot.
        for &idx in essential.iter().take(k - 1) {
            if scorers[idx].doc() < pivot {
                scorers[idx].seek(pivot);
                stats.num_seeks += 1;
            }
        }

        // Score AND advance scorers at pivot
        let mut score = 0.0f32;
        let mut any_terminated = false;
        for &idx in essential.iter() {
            if scorers[idx].doc() == pivot {
                score += idfs[idx];
                scorers[idx].advance();
                stats.num_advances += 1;
                if scorers[idx].doc() == TERMINATED {
                    any_terminated = true;
                }
            }
            // Also catch terminated scorers from seeking
            if scorers[idx].doc() == TERMINATED {
                any_terminated = true;
            }
        }

        if any_terminated {
            essential.retain(|&i| scorers[i].doc() != TERMINATED);
        }

        stats.candidates_evaluated += 1;
        stats.phase1_candidates += 1;

        // Fast path: can't beat threshold even with all non-essential terms
        let mut remaining_non_essential = *non_essential_idf_sum;
        if score + remaining_non_essential <= *threshold {
            continue;
        }

        // Lazily check non-essential terms (highest IDF first for early termination)
        for &idx in non_essential.iter().rev() {
            if score + remaining_non_essential <= *threshold {
                break;
            }
            remaining_non_essential -= idfs[idx];
            if scorers[idx].doc() > pivot {
                continue;
            }
            let doc = scorers[idx].seek(pivot);
            stats.num_seeks += 1;
            if doc == pivot {
                score += idfs[idx];
            }
        }

        if score > *threshold {
            let new_threshold = callback(pivot, score);
            stats.candidates_emitted += 1;
            if new_threshold > *threshold {
                *threshold = new_threshold;

                rebalance(essential, non_essential, non_essential_idf_sum, idfs, *threshold);
                min_essential_matches = compute_min_essential_matches(
                    essential, idfs, *non_essential_idf_sum, *threshold,
                );

                let (abs, rest) =
                    partition_absolutely_essential(essential, idfs, total_idf, *threshold);
                if !abs.is_empty() {
                    let mut remaining = rest;
                    remaining.extend_from_slice(non_essential);
                    phase2(
                        scorers, idfs, total_idf, abs, remaining, threshold, callback, stats,
                    );
                    return;
                }
            }
        }
    }
}

/// Phase 2: Intersection of absolutely essential terms + lazy scoring.
fn phase2(
    scorers: &mut [TermScorer],
    idfs: &[Score],
    total_idf: Score,
    mut abs_essential: Vec<usize>,
    mut remaining: Vec<usize>,
    threshold: &mut Score,
    callback: &mut dyn FnMut(u32, Score) -> Score,
    stats: &mut IdfPruningStats,
) {
    // Sort remaining by IDF descending for best early termination
    remaining.sort_by(|&a, &b| {
        idfs[b]
            .partial_cmp(&idfs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Find the cheapest (smallest posting list) absolutely essential scorer to drive iteration
    let driver_pos = abs_essential
        .iter()
        .enumerate()
        .min_by_key(|(_, &i)| scorers[i].size_hint())
        .map(|(pos, _)| pos)
        .unwrap();
    abs_essential.swap(0, driver_pos);

    loop {
        // Align all absolutely essential scorers to the same document
        let candidate = match align_absolutely_essential(scorers, &abs_essential, stats) {
            Some(doc) => doc,
            None => break,
        };

        // Score: sum of absolutely essential IDFs (guaranteed match)
        let abs_score: Score = abs_essential.iter().map(|&i| idfs[i]).sum();

        // Lazily score remaining terms in IDF-descending order with early termination
        let mut score = abs_score;
        let mut remaining_max: Score = remaining.iter().map(|&i| idfs[i]).sum();

        if score + remaining_max > *threshold {
            for &idx in &remaining {
                if score + remaining_max <= *threshold {
                    break;
                }
                remaining_max -= idfs[idx];
                if scorers[idx].doc() > candidate {
                    continue;
                }
                let doc = scorers[idx].seek(candidate);
                stats.num_seeks += 1;
                if doc == candidate {
                    score += idfs[idx];
                }
            }
        }

        stats.candidates_evaluated += 1;
        stats.phase2_candidates += 1;

        if score > *threshold {
            let new_threshold = callback(candidate, score);
            stats.candidates_emitted += 1;
            if new_threshold > *threshold {
                *threshold = new_threshold;
                promote_to_absolutely_essential(
                    &mut abs_essential,
                    &mut remaining,
                    idfs,
                    total_idf,
                    *threshold,
                );
            }
        }

        // Advance the driver past the candidate
        let driver = abs_essential[0];
        scorers[driver].advance();
        stats.num_advances += 1;
        if scorers[driver].doc() == TERMINATED {
            break;
        }
    }

    stats.num_absolutely_essential_at_end = abs_essential.len() as u32;
}

/// Align all absolutely essential scorers to the same document via multi-way seek.
fn align_absolutely_essential(
    scorers: &mut [TermScorer],
    abs_essential: &[usize],
    stats: &mut IdfPruningStats,
) -> Option<DocId> {
    let mut target = scorers[abs_essential[0]].doc();
    if target == TERMINATED {
        return None;
    }

    loop {
        let mut max_doc = target;
        let mut aligned = true;

        for &idx in abs_essential {
            let doc = scorers[idx].doc();
            if doc < target {
                let new_doc = scorers[idx].seek(target);
                stats.num_seeks += 1;
                if new_doc == TERMINATED {
                    return None;
                }
                if new_doc > max_doc {
                    max_doc = new_doc;
                    aligned = false;
                }
            } else if doc > max_doc {
                max_doc = doc;
                aligned = false;
            }
        }

        if aligned {
            return Some(target);
        }
        target = max_doc;
    }
}

/// Rebalance: demote essential terms whose IDF fits under the non-essential prefix sum.
fn rebalance(
    essential: &mut Vec<usize>,
    non_essential: &mut Vec<usize>,
    non_essential_idf_sum: &mut Score,
    idfs: &[Score],
    threshold: Score,
) {
    // Collect (index_in_essential, idf) and sort by IDF ascending
    let mut candidates: Vec<(usize, Score)> = essential
        .iter()
        .enumerate()
        .map(|(pos, &idx)| (pos, idfs[idx]))
        .collect();
    candidates.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut running = *non_essential_idf_sum;
    let mut to_demote: Vec<usize> = Vec::new();
    for &(pos, idf) in &candidates {
        if running + idf <= threshold {
            running += idf;
            to_demote.push(pos);
        } else {
            break;
        }
    }

    if to_demote.is_empty() {
        return;
    }

    // Remove from essential in reverse order to preserve indices
    to_demote.sort_unstable();
    for &pos in to_demote.iter().rev() {
        let idx = essential.swap_remove(pos);
        non_essential.push(idx);
    }

    *non_essential_idf_sum = running;

    // Keep non-essential sorted by IDF ascending for reverse iteration
    non_essential.sort_by(|&a, &b| {
        idfs[a]
            .partial_cmp(&idfs[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Promote remaining terms to absolutely essential if their IDF > total_idf - threshold.
fn promote_to_absolutely_essential(
    abs_essential: &mut Vec<usize>,
    remaining: &mut Vec<usize>,
    idfs: &[Score],
    total_idf: Score,
    threshold: Score,
) {
    let cutoff = total_idf - threshold;
    let mut i = 0;
    while i < remaining.len() {
        if idfs[remaining[i]] > cutoff {
            let idx = remaining.swap_remove(i);
            abs_essential.push(idx);
        } else {
            i += 1;
        }
    }
    // Re-sort remaining by IDF descending
    remaining.sort_by(|&a, &b| {
        idfs[b]
            .partial_cmp(&idfs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    use proptest::prelude::*;

    use super::*;
    use crate::fieldnorm::FieldNormReader;
    use crate::postings::SegmentPostings;
    use crate::query::Bm25Weight;
    use crate::{DocId, Score, TERMINATED};

    struct Float(Score);

    impl Eq for Float {}
    impl PartialEq for Float {
        fn eq(&self, other: &Self) -> bool {
            self.cmp(other) == Ordering::Equal
        }
    }
    impl PartialOrd for Float {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Float {
        fn cmp(&self, other: &Self) -> Ordering {
            other
                .0
                .partial_cmp(&self.0)
                .unwrap_or(Ordering::Equal)
        }
    }

    fn nearly_equals(left: Score, right: Score) -> bool {
        if left == right {
            return true;
        }
        let denom = (left.abs() + right.abs()).max(f32::EPSILON);
        (left - right).abs() / denom < 0.001
    }

    /// Create a TermScorer with IDF-only scoring from a list of doc IDs.
    fn create_idf_scorer(docs: &[DocId], idf_value: Score) -> TermScorer {
        let doc_freq = docs.len() as u64;
        if docs.is_empty() {
            let segment_postings = SegmentPostings::empty();
            let fieldnorm_reader = FieldNormReader::constant(1, 1);
            let bm25_weight = Bm25Weight::new_without_explain(idf_value, 1.0, doc_freq);
            return TermScorer::new(segment_postings, fieldnorm_reader, bm25_weight);
        }
        let max_doc = docs.iter().max().unwrap() + 1;
        let fieldnorms: Vec<u32> = vec![1; max_doc as usize];
        let doc_and_tfs: Vec<(DocId, u32)> = docs.iter().map(|&d| (d, 1)).collect();
        let bm25_weight = Bm25Weight::new_without_explain(idf_value, 1.0, doc_freq);
        let segment_postings =
            SegmentPostings::create_from_docs_and_tfs(&doc_and_tfs, Some(&fieldnorms));
        let fieldnorm_reader = FieldNormReader::for_test(&fieldnorms);
        TermScorer::new(segment_postings, fieldnorm_reader, bm25_weight)
    }

    /// Brute-force top-K: compute all (doc, score) pairs using IDF scoring, return top K.
    fn compute_topk_brute_force(
        mut scorers: Vec<TermScorer>,
        k: usize,
        idfs: &[Score],
    ) -> Vec<(DocId, Score)> {
        let mut all_docs: std::collections::BTreeMap<DocId, Score> =
            std::collections::BTreeMap::new();
        for (i, scorer) in scorers.iter_mut().enumerate() {
            while scorer.doc() != TERMINATED {
                *all_docs.entry(scorer.doc()).or_insert(0.0) += idfs[i];
                scorer.advance();
            }
        }

        let mut results: Vec<(DocId, Score)> = all_docs.into_iter().collect();
        // Sort by score descending, then by doc ascending for stability
        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        results.truncate(k);
        results
    }

    /// Top-K via idf_pruning callback.
    fn compute_topk_idf_pruning(
        scorers: Vec<TermScorer>,
        k: usize,
    ) -> Vec<(DocId, Score)> {
        let mut heap: BinaryHeap<Float> = BinaryHeap::with_capacity(k);
        let mut collected: Vec<(DocId, Score)> = Vec::new();
        let mut limit = Score::MIN;

        let callback = &mut |doc: u32, score: Score| -> Score {
            collected.push((doc, score));
            heap.push(Float(score));
            if heap.len() > k {
                heap.pop();
            }
            if heap.len() == k {
                limit = heap.peek().unwrap().0;
            }
            limit
        };

        let _stats = idf_pruning(scorers, Score::MIN, k, callback);

        // From collected, extract the final top-K
        let final_limit = if heap.len() == k {
            heap.peek().unwrap().0
        } else {
            Score::MIN
        };

        // Keep only docs that beat the final limit
        let mut results: Vec<(DocId, Score)> = collected
            .into_iter()
            .filter(|&(_, score)| score >= final_limit)
            .collect();
        // Sort by score descending, then doc ascending
        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        results.truncate(k);
        results
    }

    #[test]
    fn test_idf_pruning_basic() {
        let scorer1 = create_idf_scorer(&[0, 1, 2, 3, 4], 1.0);
        let scorer2 = create_idf_scorer(&[2, 3, 4, 5, 6], 1.5);
        let idfs = [1.0, 1.5];

        let pruning = compute_topk_idf_pruning(vec![scorer1.clone(), scorer2.clone()], 2);
        let brute = compute_topk_brute_force(vec![scorer1, scorer2], 2, &idfs);

        assert_eq!(pruning.len(), brute.len());
        for (p, b) in pruning.iter().zip(brute.iter()) {
            assert_eq!(p.0, b.0);
            assert!(nearly_equals(p.1, b.1), "{} != {}", p.1, b.1);
        }
    }

    #[test]
    fn test_idf_pruning_single_term() {
        let scorer = create_idf_scorer(&[0, 1, 2], 2.0);
        let results = compute_topk_idf_pruning(vec![scorer], 2);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_idf_pruning_no_overlap() {
        let scorer1 = create_idf_scorer(&[0, 1], 1.0);
        let scorer2 = create_idf_scorer(&[2, 3], 1.5);
        let idfs = [1.0, 1.5];

        let pruning = compute_topk_idf_pruning(vec![scorer1.clone(), scorer2.clone()], 3);
        let brute = compute_topk_brute_force(vec![scorer1, scorer2], 3, &idfs);

        assert_eq!(pruning.len(), brute.len());
        for (p, b) in pruning.iter().zip(brute.iter()) {
            assert_eq!(p.0, b.0);
            assert!(nearly_equals(p.1, b.1));
        }
    }

    #[test]
    fn test_idf_pruning_high_threshold() {
        let scorer1 = create_idf_scorer(&[0, 1, 2], 0.5);
        let scorer2 = create_idf_scorer(&[0, 1, 2], 0.5);

        let mut results: Vec<(DocId, Score)> = Vec::new();
        let _stats = idf_pruning(vec![scorer1, scorer2], 100.0, 0, &mut |doc, score| {
            results.push((doc, score));
            100.0
        });
        assert!(results.is_empty());
    }

    #[test]
    fn test_idf_pruning_empty() {
        let stats = idf_pruning(vec![], 0.0, 0, &mut |_doc, _score| 0.0);
        assert_eq!(stats.candidates_evaluated, 0);
    }

    // Property-based tests
    fn posting_list_idf(max_doc: u32) -> BoxedStrategy<Vec<DocId>> {
        (1..max_doc.max(2))
            .prop_flat_map(move |doc_freq| {
                proptest::bits::bitset::sampled(doc_freq as usize, 0..max_doc as usize)
            })
            .prop_map(|docset| docset.iter().map(|doc| doc as u32).collect::<Vec<_>>())
            .boxed()
    }

    fn gen_idf_scorers(
        num_scorers: usize,
    ) -> BoxedStrategy<(Vec<Vec<DocId>>, Vec<Score>)> {
        (10u32..80u32)
            .prop_flat_map(move |max_doc| {
                (
                    proptest::collection::vec(posting_list_idf(max_doc), num_scorers),
                    proptest::collection::vec(0.1f32..5.0f32, num_scorers),
                )
            })
            .boxed()
    }

    fn test_idf_pruning_correctness(posting_lists: &[Vec<DocId>], idf_values: &[Score]) {
        for top_k in [1, 2, 3, 5] {
            let scorers_pruning: Vec<TermScorer> = posting_lists
                .iter()
                .zip(idf_values.iter())
                .map(|(docs, &idf)| create_idf_scorer(docs, idf))
                .collect();
            let scorers_brute: Vec<TermScorer> = posting_lists
                .iter()
                .zip(idf_values.iter())
                .map(|(docs, &idf)| create_idf_scorer(docs, idf))
                .collect();

            let pruning_results = compute_topk_idf_pruning(scorers_pruning, top_k);
            let brute_results =
                compute_topk_brute_force(scorers_brute, top_k, idf_values);

            assert_eq!(
                pruning_results.len(),
                brute_results.len(),
                "top_k={top_k} posting_lists={posting_lists:?} idfs={idf_values:?}\npruning={pruning_results:?}\nbrute={brute_results:?}"
            );
            for (p, b) in pruning_results.iter().zip(brute_results.iter()) {
                assert!(
                    nearly_equals(p.1, b.1),
                    "score mismatch: {} vs {} for top_k={top_k} doc pruning={} brute={}",
                    p.1,
                    b.1,
                    p.0,
                    b.0,
                );
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]
        #[test]
        fn test_idf_pruning_two_scorers((posting_lists, idfs) in gen_idf_scorers(2)) {
            test_idf_pruning_correctness(&posting_lists, &idfs);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(500))]
        #[test]
        fn test_idf_pruning_three_scorers((posting_lists, idfs) in gen_idf_scorers(3)) {
            test_idf_pruning_correctness(&posting_lists, &idfs);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]
        #[test]
        fn test_idf_pruning_five_scorers((posting_lists, idfs) in gen_idf_scorers(5)) {
            test_idf_pruning_correctness(&posting_lists, &idfs);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]
        #[test]
        fn test_idf_pruning_ten_scorers((posting_lists, idfs) in gen_idf_scorers(10)) {
            test_idf_pruning_correctness(&posting_lists, &idfs);
        }
    }
}
