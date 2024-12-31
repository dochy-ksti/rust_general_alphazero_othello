#[derive(Debug, Clone)]
pub struct MctsArgs {
    pub temp_threshold: i32,
    pub num_mcts_sims: i32,
    pub cpuct: f32,
}

impl Default for MctsArgs {
    fn default() -> Self {
        Self {
            temp_threshold: 15,
            //temp_threshold: 100,
            num_mcts_sims: 25,
            cpuct: 1.0,
        }
    }
}
