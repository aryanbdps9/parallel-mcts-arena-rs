//! # Game Implementations Module
//!
//! This module contains implementations of all supported games in the engine.
//! Each game implements the GameState trait for compatibility with the MCTS engine.
//!
//! ## Supported Games
//! - **Othello (Reversi)**: Classic piece-flipping strategy game
//! - **Connect 4**: Gravity-based connection game
//! - **Blokus**: Polyomino tile-laying game for up to 4 players
//! - **Gomoku**: Five-in-a-row strategy game

pub mod othello;
pub mod connect4;
pub mod blokus;
pub mod gomoku;
