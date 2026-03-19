use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use crate::collector::sort_key::NaturalComparator;
use crate::collector::{SegmentSortKeyComputer, SortKeyComputer, TopNComputer};
use crate::{DocAddress, DocId, Score};

/// Runs the pruning loop for score-based top-K collection on a single segment.
///
/// Shared between `SortBySimilarityScore` and `SortBySimilarityScoreWithThreshold`
/// to avoid duplicating the alive_bitset branching logic.
fn collect_top_k_scores(
    k: usize,
    initial_threshold: Score,
    weight: &dyn crate::query::Weight,
    reader: &crate::SegmentReader,
    segment_ord: u32,
) -> crate::Result<TopNComputer<Score, DocId, NaturalComparator>> {
    let mut top_n: TopNComputer<Score, DocId, NaturalComparator> =
        TopNComputer::new_with_comparator(k, NaturalComparator);

    if initial_threshold > Score::MIN {
        top_n.threshold = Some(initial_threshold);
    }

    if let Some(alive_bitset) = reader.alive_bitset() {
        let mut threshold = initial_threshold;
        top_n.threshold = Some(threshold);
        weight.for_each_pruning(initial_threshold, reader, &mut |doc, score| {
            if alive_bitset.is_deleted(doc) {
                return threshold;
            }
            top_n.push(score, doc);
            threshold = top_n.threshold.unwrap_or(initial_threshold);
            threshold
        }, k)?;
    } else {
        weight.for_each_pruning(initial_threshold, reader, &mut |doc, score| {
            top_n.push(score, doc);
            top_n.threshold.unwrap_or(initial_threshold)
        }, k)?;
    }

    Ok(top_n)
}

/// Converts a `TopNComputer` of per-segment doc IDs into `(Score, DocAddress)` pairs.
fn into_scored_addresses(
    top_n: TopNComputer<Score, DocId, NaturalComparator>,
    segment_ord: u32,
) -> Vec<(Score, DocAddress)> {
    top_n
        .into_vec()
        .into_iter()
        .map(|cid| (cid.sort_key, DocAddress::new(segment_ord, cid.doc)))
        .collect()
}

/// Sort by similarity score.
#[derive(Clone, Debug, Copy)]
pub struct SortBySimilarityScore;

impl SortKeyComputer for SortBySimilarityScore {
    type SortKey = Score;

    type Child = SortBySimilarityScore;

    type Comparator = NaturalComparator;

    fn requires_scoring(&self) -> bool {
        true
    }

    fn segment_sort_key_computer(
        &self,
        _segment_reader: &crate::SegmentReader,
    ) -> crate::Result<Self::Child> {
        Ok(SortBySimilarityScore)
    }

    // Sorting by score is special in that it allows for the Block-Wand optimization.
    fn collect_segment_top_k(
        &self,
        k: usize,
        weight: &dyn crate::query::Weight,
        reader: &crate::SegmentReader,
        segment_ord: u32,
    ) -> crate::Result<Vec<(Self::SortKey, DocAddress)>> {
        let top_n = collect_top_k_scores(k, Score::MIN, weight, reader, segment_ord)?;
        Ok(into_scored_addresses(top_n, segment_ord))
    }
}

/// Wraps `SortBySimilarityScore` with cross-segment threshold sharing.
///
/// Uses `Arc<AtomicU32>` to share the best threshold across segments.
/// For positive f32 values, IEEE 754 bit patterns preserve ordering, so
/// `AtomicU32::fetch_max` correctly implements max for positive floats.
///
/// A global `TopNComputer` accumulates results from all completed segments,
/// so the shared threshold reflects the global K-th best score rather than
/// just the per-segment K-th best. This gives subsequent segments a much
/// tighter pruning bound.
pub(crate) struct SortBySimilarityScoreWithThreshold {
    shared_threshold: Arc<AtomicU32>,
    global_top_n: Arc<Mutex<TopNComputer<Score, DocAddress, NaturalComparator>>>,
}

impl SortBySimilarityScoreWithThreshold {
    pub fn new(k: usize) -> Self {
        // 0u32 = 0.0f32.to_bits() = "no threshold yet"
        Self {
            shared_threshold: Arc::new(AtomicU32::new(0u32)),
            global_top_n: Arc::new(Mutex::new(TopNComputer::new_with_comparator(
                k,
                NaturalComparator,
            ))),
        }
    }
}

impl SortKeyComputer for SortBySimilarityScoreWithThreshold {
    type SortKey = Score;
    type Child = SortBySimilarityScore;
    type Comparator = NaturalComparator;

    fn requires_scoring(&self) -> bool {
        true
    }

    fn segment_sort_key_computer(
        &self,
        _segment_reader: &crate::SegmentReader,
    ) -> crate::Result<Self::Child> {
        Ok(SortBySimilarityScore)
    }

    fn collect_segment_top_k(
        &self,
        k: usize,
        weight: &dyn crate::query::Weight,
        reader: &crate::SegmentReader,
        segment_ord: u32,
    ) -> crate::Result<Vec<(Self::SortKey, DocAddress)>> {
        let threshold_bits = self.shared_threshold.load(Ordering::Relaxed);
        let initial_threshold = f32::from_bits(threshold_bits);
        // Treat 0.0 as "no threshold" → use Score::MIN
        let initial_threshold = if initial_threshold > 0.0 {
            initial_threshold
        } else {
            Score::MIN
        };

        let top_n = collect_top_k_scores(k, initial_threshold, weight, reader, segment_ord)?;
        let results = into_scored_addresses(top_n, segment_ord);

        // Merge into global accumulator for better cross-segment threshold
        {
            let mut global = self.global_top_n.lock().unwrap();
            for &(score, doc_addr) in &results {
                global.push(score, doc_addr);
            }
            global.ensure_threshold();
            if let Some(global_threshold) = global.threshold {
                if global_threshold > 0.0 {
                    self.shared_threshold
                        .fetch_max(global_threshold.to_bits(), Ordering::Relaxed);
                }
            }
        }

        Ok(results)
    }
}

impl SegmentSortKeyComputer for SortBySimilarityScore {
    type SortKey = Score;
    type SegmentSortKey = Score;
    type SegmentComparator = NaturalComparator;

    #[inline(always)]
    fn segment_sort_key(&mut self, _doc: DocId, score: Score) -> Score {
        score
    }

    fn convert_segment_sort_key(&self, score: Score) -> Score {
        score
    }
}
