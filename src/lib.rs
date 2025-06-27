use rand::Rng;
use std::collections::HashMap;

pub trait GameState: Clone {
    type Move: Clone + Eq + std::hash::Hash + std::fmt::Debug;

    fn get_possible_moves(&self) -> Vec<Self::Move>;
    fn make_move(&mut self, mv: &Self::Move);
    fn is_terminal(&self) -> bool;
    fn get_winner(&self) -> Option<i32>;
    fn get_current_player(&self) -> i32;
}

struct Node<M: Clone + Eq + std::hash::Hash> {
    wins: f64,
    visits: i32,
    children: HashMap<M, Node<M>>,
}

impl<M: Clone + Eq + std::hash::Hash> Node<M> {
    fn new() -> Self {
        Node {
            wins: 0.0,
            visits: 0,
            children: HashMap::new(),
        }
    }

    fn ucb1(&self, parent_visits: i32, exploration_parameter: f64) -> f64 {
        if self.visits == 0 {
            std::f64::INFINITY
        } else {
            self.wins / self.visits as f64
                + exploration_parameter * ((parent_visits as f64).ln() / self.visits as f64).sqrt()
        }
    }
}

impl<M: Clone + Eq + std::hash::Hash> Default for Node<M> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct MCTS<S: GameState> {
    root: Node<S::Move>,
    exploration_parameter: f64,
}

impl<S: GameState> MCTS<S> {
    pub fn new(exploration_parameter: f64) -> Self {
        MCTS {
            root: Node::new(),
            exploration_parameter,
        }
    }

    pub fn advance_root(&mut self, mv: &S::Move) {
        self.root = self.root.children.remove(mv).unwrap_or_default();
    }

    pub fn get_root_children_stats(&self) -> std::collections::HashMap<S::Move, (f64, i32)> {
        self.root
            .children
            .iter()
            .map(|(mv, node)| (mv.clone(), (node.wins, node.visits)))
            .collect()
    }

    pub fn search(&mut self, state: &S, iterations: i32) -> S::Move {
        for _ in 0..iterations {
            let mut current_state = state.clone();
            let mut current_node = &mut self.root;
            let mut path = vec![];

            // Selection
            while !current_node.children.is_empty() {
                if current_state.is_terminal() {
                    break;
                }
                let moves = current_state.get_possible_moves();
                // Print moves
                // print!("[search]: Possible moves: ");
                // for movee in &moves {
                //     print!("{:?} ", movee);
                // }
                // println!();

                let best_move = moves
                    .iter()
                    .max_by(|a, b| {
                        let a_ucb = current_node
                            .children
                            .get(a)
                            .map_or(std::f64::INFINITY, |n| {
                                n.ucb1(current_node.visits, self.exploration_parameter)
                            });
                        let b_ucb = current_node
                            .children
                            .get(b)
                            .map_or(std::f64::INFINITY, |n| {
                                n.ucb1(current_node.visits, self.exploration_parameter)
                            });
                        a_ucb.partial_cmp(&b_ucb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap();

                path.push(best_move.clone());
                current_state.make_move(best_move);
                current_node = current_node.children.entry(best_move.clone()).or_insert_with(Node::new);
            }

            // Expansion
            if !current_state.is_terminal() {
                let moves = current_state.get_possible_moves();
                for mv in moves {
                    current_node.children.insert(mv, Node::new());
                }
            }

            // Simulation
            let mut winner = current_state.get_winner();
            while !current_state.is_terminal() {
                let moves = current_state.get_possible_moves();
                if moves.is_empty() {
                    break;
                }
                let mv = moves[rand::thread_rng().gen_range(0..moves.len())].clone();
                current_state.make_move(&mv);
                winner = current_state.get_winner();
            }

            // Backpropagation
            self.root.visits += 1;
            let mut traversal_state = state.clone();
            let mut node_to_update = &mut self.root;
            for mv in path {
                let parent_player = traversal_state.get_current_player();
                traversal_state.make_move(&mv);
                node_to_update = node_to_update.children.get_mut(&mv).unwrap();
                node_to_update.visits += 1;
                if let Some(w) = winner {
                    if w == parent_player {
                        node_to_update.wins += 1.0;
                    } else if w == -1 {
                        node_to_update.wins += 0.5;
                    }
                }
            }
        }

        let best_move = self.root.children.iter()
            .max_by_key(|(_, node)| node.visits)
            .map(|(mv, _)| mv.clone())
            .unwrap();

        best_move
    }
}
