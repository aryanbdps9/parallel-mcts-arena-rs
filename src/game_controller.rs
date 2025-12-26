//! # Game Controller Module - Central Game State Management
//!
//! This module provides the `GameController` which serves as the single source of truth
//! for the authoritative game state. It ensures proper separation between:
//!
//! - **Authoritative Game State**: The "real" game state owned by the controller
//! - **AI Search Trees' Game States**: Clones used during MCTS exploration  
//! - **UI Render States**: Copies used for display purposes
//!
//! ## Architecture Overview
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                       GameController                                │
//! │  ┌─────────────────────────────────────────────────────────────┐    │
//! │  │              Authoritative Game State                       │    │
//! │  │  • Single source of truth                                   │    │
//! │  │  • All moves validated here before application              │    │
//! │  │  • Move history maintained                                  │    │
//! │  └─────────────────────────────────────────────────────────────┘    │
//! │                           │                                         │
//! │              ┌────────────┼────────────┐                            │
//! │              ▼            ▼            ▼                            │
//! │  ┌───────────────┐ ┌───────────┐ ┌─────────────────┐                │
//! │  │ AI Worker     │ │ UI Thread │ │ Event Handler   │                │
//! │  │ (cloned state)│ │ (view)    │ │ (requests)      │                │
//! │  └───────────────┘ └───────────┘ └─────────────────┘                │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Benefits
//! - **Thread Safety**: Clear ownership model prevents race conditions
//! - **Move Validation**: All moves are validated before application
//! - **Consistency**: Single source of truth prevents state divergence
//! - **Auditability**: Complete move history with timestamps

use crate::game_wrapper::{GameWrapper, MoveWrapper};
use mcts::GameState;
use std::time::SystemTime;

/// Result of attempting to apply a move
#[derive(Debug, Clone)]
pub enum MoveResult {
    /// Move was successfully applied
    Success {
        /// The applied move
        move_made: MoveWrapper,
        /// Player who made the move
        player: i32,
        /// Whether the game is now over
        game_over: bool,
        /// Winner if game is over (None for draw)
        winner: Option<i32>,
    },
    /// Move was rejected as invalid
    Invalid {
        /// Reason the move was rejected
        reason: MoveValidationError,
    },
    /// Game is already over, no more moves allowed
    GameOver,
}

/// Errors that can occur during move validation
#[derive(Debug, Clone)]
pub enum MoveValidationError {
    /// Move is not in the list of legal moves
    IllegalMove,
    /// Move type doesn't match the current game
    MismatchedGameType,
    /// The game is already in a terminal state
    GameAlreadyOver,
    /// Custom validation error with message
    Custom(String),
}

impl std::fmt::Display for MoveValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MoveValidationError::IllegalMove => write!(f, "Illegal move"),
            MoveValidationError::MismatchedGameType => write!(f, "Move type doesn't match game"),
            MoveValidationError::GameAlreadyOver => write!(f, "Game is already over"),
            MoveValidationError::Custom(msg) => write!(f, "{}", msg),
        }
    }
}

/// A single entry in the move history
#[derive(Debug, Clone)]
pub struct MoveHistoryEntry {
    /// When the move was made
    pub timestamp: SystemTime,
    /// Player who made the move
    pub player: i32,
    /// The move that was made
    pub move_made: MoveWrapper,
    /// Move number (1-indexed)
    pub move_number: usize,
}

impl MoveHistoryEntry {
    /// Create a new move history entry
    pub fn new(player: i32, move_made: MoveWrapper, move_number: usize) -> Self {
        Self {
            timestamp: SystemTime::now(),
            player,
            move_made,
            move_number,
        }
    }
}

/// Current game status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameStatus {
    /// Game is still in progress
    InProgress,
    /// Game ended with a winner
    Win(i32),
    /// Game ended in a draw
    Draw,
}

impl GameStatus {
    /// Check if the game is over
    pub fn is_game_over(&self) -> bool {
        !matches!(self, GameStatus::InProgress)
    }
}

/// The central game controller that owns the authoritative game state
///
/// This is the single source of truth for the game state. All moves must
/// go through the controller, which validates them before application.
///
/// # Usage
/// ```rust,ignore
/// let mut controller = GameController::new(GameWrapper::Gomoku(...));
///
/// // Attempt to make a move
/// match controller.try_make_move(move_wrapper) {
///     MoveResult::Success { game_over, winner, .. } => {
///         // Move was applied
///     }
///     MoveResult::Invalid { reason } => {
///         // Move was rejected
///     }
///     MoveResult::GameOver => {
///         // Can't move, game is over
///     }
/// }
///
/// // Get a clone for AI to search
/// let state_for_ai = controller.get_state_for_search();
///
/// // Get current state for UI to render
/// let ui_state = controller.get_render_state();
/// ```
#[derive(Debug, Clone)]
pub struct GameController {
    /// The authoritative game state
    game_state: GameWrapper,
    /// Complete history of moves made
    move_history: Vec<MoveHistoryEntry>,
    /// Current game status
    status: GameStatus,
}

impl GameController {
    /// Create a new game controller with the given initial state
    pub fn new(initial_state: GameWrapper) -> Self {
        Self {
            game_state: initial_state,
            move_history: Vec::new(),
            status: GameStatus::InProgress,
        }
    }

    /// Validate a move without applying it
    ///
    /// Returns Ok(()) if the move is valid, or an error describing why it's invalid.
    pub fn validate_move(&self, mv: &MoveWrapper) -> Result<(), MoveValidationError> {
        // Check if game is already over
        if self.status.is_game_over() {
            return Err(MoveValidationError::GameAlreadyOver);
        }

        // Check if the move is legal according to the game rules
        if !self.game_state.is_legal(mv) {
            return Err(MoveValidationError::IllegalMove);
        }

        Ok(())
    }

    /// Attempt to make a move
    ///
    /// Validates the move and applies it if valid. Returns the result of the attempt.
    pub fn try_make_move(&mut self, mv: MoveWrapper) -> MoveResult {
        // Validate the move first
        if let Err(reason) = self.validate_move(&mv) {
            return MoveResult::Invalid { reason };
        }

        // Get the current player before making the move
        let player = self.game_state.get_current_player();
        let move_number = self.move_history.len() + 1;

        // Apply the move
        self.game_state.make_move(&mv);

        // Record in history
        self.move_history.push(MoveHistoryEntry::new(player, mv.clone(), move_number));

        // Check for game over
        let game_over = self.game_state.is_terminal();
        let winner = if game_over {
            self.game_state.get_winner()
        } else {
            None
        };

        // Update status
        if game_over {
            self.status = match winner {
                Some(w) => GameStatus::Win(w),
                None => GameStatus::Draw,
            };
        }

        MoveResult::Success {
            move_made: mv,
            player,
            game_over,
            winner,
        }
    }

    /// Force a move without validation (for AI moves that are trusted)
    ///
    /// Use with caution - this bypasses validation. Should only be used
    /// for moves that come from the AI search which uses the same game rules.
    pub fn apply_trusted_move(&mut self, mv: MoveWrapper) -> MoveResult {
        if self.status.is_game_over() {
            return MoveResult::GameOver;
        }

        let player = self.game_state.get_current_player();
        let move_number = self.move_history.len() + 1;

        self.game_state.make_move(&mv);
        self.move_history.push(MoveHistoryEntry::new(player, mv.clone(), move_number));

        let game_over = self.game_state.is_terminal();
        let winner = if game_over {
            self.game_state.get_winner()
        } else {
            None
        };

        if game_over {
            self.status = match winner {
                Some(w) => GameStatus::Win(w),
                None => GameStatus::Draw,
            };
        }

        MoveResult::Success {
            move_made: mv,
            player,
            game_over,
            winner,
        }
    }

    /// Get a clone of the game state for AI to search
    ///
    /// The returned state can be freely modified by the AI without
    /// affecting the authoritative state.
    pub fn get_state_for_search(&self) -> GameWrapper {
        self.game_state.clone()
    }

    /// Get a reference to the game state for rendering
    ///
    /// This should be used for display purposes only. Do not store
    /// this reference across frames.
    pub fn get_render_state(&self) -> &GameWrapper {
        &self.game_state
    }

    /// Get the current player
    pub fn get_current_player(&self) -> i32 {
        self.game_state.get_current_player()
    }

    /// Get the current game status
    pub fn get_status(&self) -> GameStatus {
        self.status
    }

    /// Check if the game is over
    pub fn is_game_over(&self) -> bool {
        self.status.is_game_over()
    }

    /// Get the winner if the game is over
    pub fn get_winner(&self) -> Option<i32> {
        match self.status {
            GameStatus::Win(w) => Some(w),
            _ => None,
        }
    }

    /// Get the complete move history
    pub fn get_move_history(&self) -> &[MoveHistoryEntry] {
        &self.move_history
    }

    /// Get the number of moves made
    pub fn move_count(&self) -> usize {
        self.move_history.len()
    }

    /// Get the last move made, if any
    pub fn get_last_move(&self) -> Option<&MoveHistoryEntry> {
        self.move_history.last()
    }

    /// Get the board for rendering
    pub fn get_board(&self) -> &Vec<Vec<i32>> {
        self.game_state.get_board()
    }

    /// Get legal moves for the current player
    pub fn get_legal_moves(&self) -> Vec<MoveWrapper> {
        if self.status.is_game_over() {
            Vec::new()
        } else {
            self.game_state.get_possible_moves()
        }
    }

    /// Reset the game to its initial state
    pub fn reset(&mut self, new_state: GameWrapper) {
        self.game_state = new_state;
        self.move_history.clear();
        self.status = GameStatus::InProgress;
    }

    /// Format move history as a string suitable for copying to clipboard
    pub fn format_history_for_clipboard(&self) -> String {
        if self.move_history.is_empty() {
            return String::from("No moves made yet.");
        }

        let game_name = match &self.game_state {
            GameWrapper::Gomoku(_) => "Gomoku",
            GameWrapper::Connect4(_) => "Connect 4",
            GameWrapper::Othello(_) => "Othello",
            GameWrapper::Blokus(_) => "Blokus",
            GameWrapper::Hive(_) => "Hive",
        };

        let mut output = format!("=== {} Game History ===\n\n", game_name);

        for entry in &self.move_history {
            let player_name = self.get_player_name(entry.player);
            output.push_str(&format!(
                "{}. {} - {}\n",
                entry.move_number,
                player_name,
                entry.move_made
            ));
        }

        // Add game result if over
        match self.status {
            GameStatus::Win(winner) => {
                let winner_name = self.get_player_name(winner);
                output.push_str(&format!("\nResult: {} wins!\n", winner_name));
            }
            GameStatus::Draw => {
                output.push_str("\nResult: Draw\n");
            }
            GameStatus::InProgress => {
                output.push_str(&format!("\n(Game in progress - {} to move)\n", 
                    self.get_player_name(self.get_current_player())));
            }
        }

        output
    }

    /// Get a human-readable player name
    fn get_player_name(&self, player_id: i32) -> String {
        match &self.game_state {
            GameWrapper::Blokus(_) => {
                match player_id {
                    1 => "Blue".to_string(),
                    2 => "Yellow".to_string(),
                    3 => "Red".to_string(),
                    4 => "Green".to_string(),
                    _ => format!("Player {}", player_id),
                }
            }
            GameWrapper::Othello(_) => {
                if player_id == 1 { "Black".to_string() } else { "White".to_string() }
            }
            _ => {
                if player_id == 1 { "Player 1".to_string() } else { "Player 2".to_string() }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::gomoku::GomokuState;

    #[test]
    fn test_valid_move() {
        let state = GameWrapper::Gomoku(GomokuState::new(15, 5));
        let mut controller = GameController::new(state);

        let mv = MoveWrapper::Gomoku(crate::games::gomoku::GomokuMove(7, 7));
        match controller.try_make_move(mv) {
            MoveResult::Success { player, game_over, .. } => {
                assert_eq!(player, 1);
                assert!(!game_over);
            }
            _ => panic!("Expected successful move"),
        }
    }

    #[test]
    fn test_invalid_move_occupied() {
        let state = GameWrapper::Gomoku(GomokuState::new(15, 5));
        let mut controller = GameController::new(state);

        // Make first move
        let mv1 = MoveWrapper::Gomoku(crate::games::gomoku::GomokuMove(7, 7));
        controller.try_make_move(mv1);

        // Try to make same move again
        let mv2 = MoveWrapper::Gomoku(crate::games::gomoku::GomokuMove(7, 7));
        match controller.try_make_move(mv2) {
            MoveResult::Invalid { reason: MoveValidationError::IllegalMove } => {}
            _ => panic!("Expected illegal move error"),
        }
    }

    #[test]
    fn test_move_history() {
        let state = GameWrapper::Gomoku(GomokuState::new(15, 5));
        let mut controller = GameController::new(state);

        let mv1 = MoveWrapper::Gomoku(crate::games::gomoku::GomokuMove(7, 7));
        let mv2 = MoveWrapper::Gomoku(crate::games::gomoku::GomokuMove(7, 8));

        controller.try_make_move(mv1);
        controller.try_make_move(mv2);

        assert_eq!(controller.move_count(), 2);
        assert_eq!(controller.get_move_history()[0].player, 1);
        assert_eq!(controller.get_move_history()[1].player, -1);
    }

    #[test]
    fn test_reset() {
        let state = GameWrapper::Gomoku(GomokuState::new(15, 5));
        let mut controller = GameController::new(state);

        let mv = MoveWrapper::Gomoku(crate::games::gomoku::GomokuMove(7, 7));
        controller.try_make_move(mv);
        assert_eq!(controller.move_count(), 1);

        let new_state = GameWrapper::Gomoku(GomokuState::new(15, 5));
        controller.reset(new_state);
        
        assert_eq!(controller.move_count(), 0);
        assert!(matches!(controller.status, GameStatus::InProgress));
    }

    #[test]
    fn test_format_history() {
        let state = GameWrapper::Gomoku(GomokuState::new(15, 5));
        let mut controller = GameController::new(state);

        let mv = MoveWrapper::Gomoku(crate::games::gomoku::GomokuMove(7, 7));
        controller.try_make_move(mv);

        let history = controller.format_history_for_clipboard();
        assert!(history.contains("Gomoku Game History"));
        assert!(history.contains("1. Player 1 - G(7,7)"));
    }
}
