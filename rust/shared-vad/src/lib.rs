use earshot::Detector;

const EARSHOT_FRAME: usize = 256; // 16ms at 16kHz
const EARSHOT_MS: usize = 16;

pub struct Vad {
    detector: Detector,
    probabilities: Vec<f32>, // for earshot-native 16ms chunks
    leftovers: Vec<i16>,     // samples not yet divisible by 16ms
}

impl Default for Vad {
    fn default() -> Self {
        Self::new()
    }
}

impl Vad {
    pub fn new() -> Self {
        Self {
            detector: Detector::default(),
            probabilities: Vec::new(),
            leftovers: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.detector = Detector::default();
        self.probabilities.clear();
        self.leftovers.clear();
    }

    pub fn consume(&mut self, samples: &[i16]) {
        let mut pos = 0;

        if !self.leftovers.is_empty() {
            let need = EARSHOT_FRAME - self.leftovers.len();
            if samples.len() < need {
                self.leftovers.extend_from_slice(samples);
                return; // still not enough
            }
            self.leftovers.extend_from_slice(&samples[..need]);
            self.probabilities
                .push(self.detector.predict_i16(&self.leftovers));
            self.leftovers.clear();
            pos = need;
        }

        while pos + EARSHOT_FRAME <= samples.len() {
            let chunk = &samples[pos..(pos + EARSHOT_FRAME)];
            self.probabilities.push(self.detector.predict_i16(chunk));
            pos += EARSHOT_FRAME;
        }

        if pos < samples.len() {
            self.leftovers.extend_from_slice(&samples[pos..]);
        }
    }

    pub fn probability_at_cs(&self, cs: i64) -> f32 {
        let t_ms = cs as f32 * 10.0;
        let probabilities_pos: f32 = t_ms / (EARSHOT_MS as f32);

        if self.probabilities.is_empty() {
            return 0.0;
        }

        if probabilities_pos <= 0.0 {
            return self.probabilities[0];
        }

        let lo = probabilities_pos.floor() as usize;
        if lo >= self.probabilities.len() - 1 {
            return self.end_p();
        }
        let lo_val = self.probabilities[lo];
        let hi_val = self.probabilities[lo + 1];
        let hi_weight = probabilities_pos - lo as f32;
        hi_val * hi_weight + lo_val * (1. - hi_weight)
    }

    pub fn end_p(&self) -> f32 {
        self.probabilities.last().copied().unwrap_or(0.0)
    }

    pub fn end_cs(&self) -> i64 {
        let t_ms = self.probabilities.len() * EARSHOT_MS;
        (t_ms as f32 / 10.).floor() as i64
    }
}
