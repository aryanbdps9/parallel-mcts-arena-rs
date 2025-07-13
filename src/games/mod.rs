//! # Game Implementations Module
//!
//! This module contains implementations of all supported games in the MCTS engine.
//! Each game implements the `GameState` trait to provide a consistent interface
//! for the Monte Carlo Tree Search algorithm and user interface.
//!
//! ## Supported Games
//! - **Othello (Reversi)**: Classic 8x8 piece-flipping strategy game for 2 players
//! - **Connect 4**: Gravity-based connection game on a 6x7 grid for 2 players  
//! - **Blokus**: Polyomino tile-laying strategy game for 2-4 players on a 20x20 board
//! - **Gomoku (Five in a Row)**: Configurable N-in-a-row game on variable board sizes
//!
//! ## Game Trait Implementation
//! All games implement the `mcts::GameState` trait which provides:
//! - Move generation and validation
//! - State transitions and game rules
//! - Terminal state detection and winner determination
//! - Board representation and current player tracking
//!
//! ## Adding New Games
//! To add a new game, create a new module and implement:
//! 1. A move type (typically a struct with coordinates)
//! 2. A game state type with the GameState trait
//! 3. Display and parsing implementations for moves
//! 4. Game-specific rules and win conditions

pub mod blokus;
pub mod connect4;
pub mod gomoku;
pub mod othello;
