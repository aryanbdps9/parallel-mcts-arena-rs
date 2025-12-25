//! # Hive Game Implementation
//!
//! This module implements the Hive board game, a strategic two-player game
//! where players place and move hexagonal tiles representing insects.
//!
//! ## Game Overview
//! Hive is played without a board - pieces are placed adjacent to each other
//! on a flat surface. Each piece type has unique movement abilities:
//! - **Queen Bee**: Moves one space; must be placed by turn 4
//! - **Beetle**: Moves one space; can climb on top of the hive
//! - **Spider**: Moves exactly three spaces around the hive
//! - **Grasshopper**: Jumps in a straight line over pieces
//! - **Ant**: Moves any number of spaces around the hive
//!
//! ## Rules
//! - First player places any piece; second player places adjacent to it
//! - After that, pieces must be placed touching only own color
//! - Queen must be placed by turn 4 (each player's 4th piece)
//! - Movement is only allowed after Queen is placed
//! - The Hive must remain connected at all times (One Hive rule)
//! - A player wins by completely surrounding opponent's Queen
//!
//! ## Coordinate System
//! Uses axial coordinates (q, r) for hexagonal grid representation.
//! The hexagons use "pointy-top" orientation.

use crate::GameState;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

/// Piece types in Hive
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PieceType {
    Queen,       // 1 per player - moves 1 space
    Beetle,      // 2 per player - moves 1 space, can climb
    Spider,      // 2 per player - moves exactly 3 spaces
    Grasshopper, // 3 per player - jumps in straight line
    Ant,         // 3 per player - moves any number of spaces
}

impl PieceType {
    /// Get the count of each piece type per player
    pub fn count_per_player(&self) -> usize {
        match self {
            PieceType::Queen => 1,
            PieceType::Beetle => 2,
            PieceType::Spider => 2,
            PieceType::Grasshopper => 3,
            PieceType::Ant => 3,
        }
    }

    /// Get all piece types
    pub fn all() -> &'static [PieceType] {
        &[
            PieceType::Queen,
            PieceType::Beetle,
            PieceType::Spider,
            PieceType::Grasshopper,
            PieceType::Ant,
        ]
    }

    /// Get a single-character representation of the piece
    pub fn char(&self) -> char {
        match self {
            PieceType::Queen => 'Q',
            PieceType::Beetle => 'B',
            PieceType::Spider => 'S',
            PieceType::Grasshopper => 'G',
            PieceType::Ant => 'A',
        }
    }
}

/// A piece in Hive with its owner
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Piece {
    pub piece_type: PieceType,
    pub player: i32, // 1 or -1
}

impl Piece {
    pub fn new(piece_type: PieceType, player: i32) -> Self {
        Self { piece_type, player }
    }
}

/// Axial coordinates for hexagonal grid (pointy-top orientation)
/// 
/// Neighbors in axial coords:
/// - (+1,  0): East
/// - (-1,  0): West
/// - ( 0, +1): Southeast
/// - ( 0, -1): Northwest
/// - (+1, -1): Northeast
/// - (-1, +1): Southwest
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HexCoord {
    pub q: i32,
    pub r: i32,
}

impl HexCoord {
    pub fn new(q: i32, r: i32) -> Self {
        Self { q, r }
    }

    /// Get all 6 neighboring hex coordinates
    pub fn neighbors(&self) -> [HexCoord; 6] {
        [
            HexCoord::new(self.q + 1, self.r),     // E
            HexCoord::new(self.q - 1, self.r),     // W
            HexCoord::new(self.q, self.r + 1),     // SE
            HexCoord::new(self.q, self.r - 1),     // NW
            HexCoord::new(self.q + 1, self.r - 1), // NE
            HexCoord::new(self.q - 1, self.r + 1), // SW
        ]
    }

    /// Get neighbor in a specific direction (0-5)
    pub fn neighbor(&self, direction: usize) -> HexCoord {
        self.neighbors()[direction % 6]
    }

    /// Get direction offsets
    pub fn direction_offsets() -> [(i32, i32); 6] {
        [
            (1, 0),   // E
            (-1, 0),  // W
            (0, 1),   // SE
            (0, -1),  // NW
            (1, -1),  // NE
            (-1, 1),  // SW
        ]
    }
}

/// Represents a move in Hive
///
/// A move is either placing a new piece or moving an existing piece.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum HiveMove {
    /// Place a new piece at a position
    Place {
        piece_type: PieceType,
        to: HexCoord,
    },
    /// Move an existing piece from one position to another
    Move {
        from: HexCoord,
        to: HexCoord,
    },
    /// Pass turn (only valid when no moves available but game continues)
    Pass,
}

impl fmt::Display for HiveMove {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HiveMove::Place { piece_type, to } => {
                write!(f, "{}({},{})", piece_type.char(), to.q, to.r)
            }
            HiveMove::Move { from, to } => {
                write!(f, "({},{})->({},{})", from.q, from.r, to.q, to.r)
            }
            HiveMove::Pass => write!(f, "Pass"),
        }
    }
}

/// Represents the complete state of a Hive game
#[derive(Debug, Clone)]
pub struct HiveState {
    /// The board: maps hex coordinates to stacks of pieces (bottom to top)
    /// A stack can have multiple pieces when beetles climb on top
    board: HashMap<HexCoord, Vec<Piece>>,
    
    /// Current player (1 or -1)
    current_player: i32,
    
    /// Pieces remaining in each player's hand
    /// Maps (player, piece_type) -> count remaining
    hands: HashMap<(i32, PieceType), usize>,
    
    /// Turn number (starts at 1)
    turn: usize,
    
    /// Number of pieces each player has placed
    pieces_placed: [usize; 2],
    
    /// Whether each player's queen is on the board
    queen_placed: [bool; 2],
    
    /// Last move made
    last_move: Option<HiveMove>,
    
    /// Cached board for GameState trait (updated lazily)
    cached_board: Vec<Vec<i32>>,
    cached_board_valid: bool,
}

impl HiveState {
    /// Create a new Hive game
    pub fn new() -> Self {
        let mut hands = HashMap::new();
        
        // Initialize hands for both players
        for player in [1, -1] {
            for piece_type in PieceType::all() {
                hands.insert((player, *piece_type), piece_type.count_per_player());
            }
        }
        
        Self {
            board: HashMap::new(),
            current_player: 1,
            hands,
            turn: 1,
            pieces_placed: [0, 0],
            queen_placed: [false, false],
            last_move: None,
            cached_board: vec![vec![0; 21]; 21],
            cached_board_valid: false,
        }
    }

    /// Get player index (0 for player 1, 1 for player -1)
    fn player_index(player: i32) -> usize {
        if player == 1 { 0 } else { 1 }
    }

    /// Get the top piece at a position (if any)
    pub fn get_top_piece(&self, coord: &HexCoord) -> Option<&Piece> {
        self.board.get(coord).and_then(|stack| stack.last())
    }

    /// Get the full stack at a position
    pub fn get_stack(&self, coord: &HexCoord) -> Option<&Vec<Piece>> {
        self.board.get(coord)
    }

    /// Check if a position has any pieces
    pub fn is_occupied(&self, coord: &HexCoord) -> bool {
        self.board.get(coord).map(|s| !s.is_empty()).unwrap_or(false)
    }

    /// Get all occupied positions
    pub fn occupied_positions(&self) -> impl Iterator<Item = &HexCoord> {
        self.board.iter()
            .filter(|(_, stack)| !stack.is_empty())
            .map(|(coord, _)| coord)
    }

    /// Get count of remaining pieces in hand
    pub fn pieces_in_hand(&self, player: i32, piece_type: PieceType) -> usize {
        *self.hands.get(&(player, piece_type)).unwrap_or(&0)
    }

    /// Check if player's queen is placed
    pub fn is_queen_placed(&self, player: i32) -> bool {
        self.queen_placed[Self::player_index(player)]
    }

    /// Get positions adjacent to the hive (empty positions with at least one neighbor)
    fn get_adjacent_empty_positions(&self) -> HashSet<HexCoord> {
        let mut adjacent = HashSet::new();
        for coord in self.occupied_positions() {
            for neighbor in coord.neighbors() {
                if !self.is_occupied(&neighbor) {
                    adjacent.insert(neighbor);
                }
            }
        }
        adjacent
    }

    /// Get valid placement positions for current player
    fn get_placement_positions(&self) -> Vec<HexCoord> {
        // First move: place anywhere
        if self.board.is_empty() {
            return vec![HexCoord::new(0, 0)];
        }
        
        // Second move (first move of second player): must be adjacent to existing piece
        if self.turn == 2 {
            return self.get_adjacent_empty_positions().into_iter().collect();
        }
        
        // Normal placement: adjacent to own pieces, not adjacent to opponent pieces
        let mut valid = Vec::new();
        let adjacent_empty = self.get_adjacent_empty_positions();
        
        for pos in adjacent_empty {
            let mut touches_own = false;
            let mut touches_opponent = false;
            
            for neighbor in pos.neighbors() {
                if let Some(top) = self.get_top_piece(&neighbor) {
                    if top.player == self.current_player {
                        touches_own = true;
                    } else {
                        touches_opponent = true;
                    }
                }
            }
            
            if touches_own && !touches_opponent {
                valid.push(pos);
            }
        }
        
        valid
    }

    /// Check if removing a piece would disconnect the hive (One Hive rule)
    fn would_break_hive(&self, coord: &HexCoord) -> bool {
        // If there's a stack with more than one piece, removing top doesn't break hive
        if let Some(stack) = self.board.get(coord) {
            if stack.len() > 1 {
                return false;
            }
        }
        
        // Get all occupied positions except the one being removed
        let occupied: Vec<HexCoord> = self.occupied_positions()
            .filter(|&c| c != coord)
            .cloned()
            .collect();
        
        if occupied.is_empty() {
            return false; // Only one piece, can't break
        }
        
        // BFS to check connectivity
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(occupied[0]);
        visited.insert(occupied[0]);
        
        while let Some(current) = queue.pop_front() {
            for neighbor in current.neighbors() {
                if occupied.contains(&neighbor) && !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }
        
        visited.len() != occupied.len()
    }

    /// Check if a piece can physically slide from one position to an adjacent one
    /// (the "freedom to move" rule - can't squeeze through tight gaps)
    fn can_slide(&self, from: &HexCoord, to: &HexCoord) -> bool {
        // Find which neighbors are shared between from and to
        let from_neighbors: HashSet<HexCoord> = from.neighbors().into_iter().collect();
        let to_neighbors: HashSet<HexCoord> = to.neighbors().into_iter().collect();
        
        // The two positions that are adjacent to both from and to
        let shared: Vec<HexCoord> = from_neighbors.intersection(&to_neighbors).cloned().collect();
        
        if shared.len() != 2 {
            return true; // Something's wrong with geometry, allow move
        }
        
        // Can slide if at least one of the shared positions is empty
        !self.is_occupied(&shared[0]) || !self.is_occupied(&shared[1])
    }

    /// Get all valid moves for the Queen Bee (one space slide)
    fn get_queen_moves(&self, from: &HexCoord) -> Vec<HexCoord> {
        if self.would_break_hive(from) {
            return vec![];
        }
        
        let mut moves = Vec::new();
        for neighbor in from.neighbors() {
            // Must slide to empty space
            if !self.is_occupied(&neighbor) {
                // Must remain connected to hive
                let has_hive_neighbor = neighbor.neighbors().iter()
                    .any(|n| n != from && self.is_occupied(n));
                
                if has_hive_neighbor && self.can_slide(from, &neighbor) {
                    moves.push(neighbor);
                }
            }
        }
        moves
    }

    /// Get all valid moves for the Beetle (one space, can climb)
    fn get_beetle_moves(&self, from: &HexCoord) -> Vec<HexCoord> {
        if self.would_break_hive(from) {
            return vec![];
        }
        
        let mut moves = Vec::new();
        let from_height = self.board.get(from).map(|s| s.len()).unwrap_or(0);
        
        for neighbor in from.neighbors() {
            let to_height = self.board.get(&neighbor).map(|s| s.len()).unwrap_or(0);
            
            // Beetle can always climb on/off the hive
            if from_height > 1 || to_height > 0 {
                // On top of hive or climbing onto hive - can always move
                moves.push(neighbor);
            } else if !self.is_occupied(&neighbor) {
                // Ground level movement - must stay connected and can slide
                let has_hive_neighbor = neighbor.neighbors().iter()
                    .any(|n| n != from && self.is_occupied(n));
                
                if has_hive_neighbor && self.can_slide(from, &neighbor) {
                    moves.push(neighbor);
                }
            }
        }
        moves
    }

    /// Get all valid moves for the Spider (exactly 3 spaces)
    fn get_spider_moves(&self, from: &HexCoord) -> Vec<HexCoord> {
        if self.would_break_hive(from) {
            return vec![];
        }
        
        // BFS for exactly 3 steps
        let mut visited_at_step: Vec<HashSet<HexCoord>> = vec![HashSet::new(); 4];
        visited_at_step[0].insert(*from);
        
        for step in 0..3 {
            for pos in visited_at_step[step].clone() {
                for neighbor in pos.neighbors() {
                    // Must be empty
                    if self.is_occupied(&neighbor) || neighbor == *from {
                        continue;
                    }
                    
                    // Must stay connected to hive
                    let has_hive_neighbor = neighbor.neighbors().iter()
                        .any(|n| n != from && self.is_occupied(n));
                    
                    if !has_hive_neighbor {
                        continue;
                    }
                    
                    // Can slide
                    if !self.can_slide(&pos, &neighbor) {
                        continue;
                    }
                    
                    // Haven't visited this in an earlier step
                    let already_visited = (0..=step).any(|s| visited_at_step[s].contains(&neighbor));
                    if !already_visited {
                        visited_at_step[step + 1].insert(neighbor);
                    }
                }
            }
        }
        
        visited_at_step[3].iter().cloned().collect()
    }

    /// Get all valid moves for the Grasshopper (jump in straight line)
    fn get_grasshopper_moves(&self, from: &HexCoord) -> Vec<HexCoord> {
        if self.would_break_hive(from) {
            return vec![];
        }
        
        let mut moves = Vec::new();
        
        for (dq, dr) in HexCoord::direction_offsets() {
            let mut current = HexCoord::new(from.q + dq, from.r + dr);
            
            // Must jump over at least one piece
            if !self.is_occupied(&current) {
                continue;
            }
            
            // Keep going until we find an empty space
            while self.is_occupied(&current) {
                current = HexCoord::new(current.q + dq, current.r + dr);
            }
            
            moves.push(current);
        }
        
        moves
    }

    /// Get all valid moves for the Ant (any number of spaces around the hive)
    fn get_ant_moves(&self, from: &HexCoord) -> Vec<HexCoord> {
        if self.would_break_hive(from) {
            return vec![];
        }
        
        // BFS to find all reachable empty spaces
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        
        // Start with immediate neighbors
        for neighbor in from.neighbors() {
            if !self.is_occupied(&neighbor) {
                let has_hive_neighbor = neighbor.neighbors().iter()
                    .any(|n| n != from && self.is_occupied(n));
                
                if has_hive_neighbor && self.can_slide(from, &neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }
        
        while let Some(current) = queue.pop_front() {
            for neighbor in current.neighbors() {
                if visited.contains(&neighbor) || neighbor == *from {
                    continue;
                }
                
                if self.is_occupied(&neighbor) {
                    continue;
                }
                
                let has_hive_neighbor = neighbor.neighbors().iter()
                    .any(|n| n != from && self.is_occupied(n));
                
                if has_hive_neighbor && self.can_slide(&current, &neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }
        
        visited.into_iter().collect()
    }

    /// Get all movement moves for a piece at a position
    fn get_piece_moves(&self, coord: &HexCoord) -> Vec<HexCoord> {
        let piece = match self.get_top_piece(coord) {
            Some(p) => p,
            None => return vec![],
        };
        
        // Can only move own pieces
        if piece.player != self.current_player {
            return vec![];
        }
        
        match piece.piece_type {
            PieceType::Queen => self.get_queen_moves(coord),
            PieceType::Beetle => self.get_beetle_moves(coord),
            PieceType::Spider => self.get_spider_moves(coord),
            PieceType::Grasshopper => self.get_grasshopper_moves(coord),
            PieceType::Ant => self.get_ant_moves(coord),
        }
    }

    /// Check if a player's queen is surrounded (game over condition)
    fn is_queen_surrounded(&self, player: i32) -> bool {
        // Find queen position
        for (coord, stack) in &self.board {
            for piece in stack {
                if piece.player == player && piece.piece_type == PieceType::Queen {
                    // Check if all 6 neighbors are occupied
                    return coord.neighbors().iter()
                        .all(|n| self.is_occupied(n));
                }
            }
        }
        false
    }

    /// Get the line size (not applicable to Hive)
    pub fn get_line_size(&self) -> usize {
        1
    }

    /// Get last move coordinates
    pub fn get_last_move(&self) -> Option<Vec<(usize, usize)>> {
        // Convert last move to approximate grid coordinates for highlighting
        match &self.last_move {
            Some(HiveMove::Place { to, .. }) => {
                Some(vec![((to.q + 10) as usize, (to.r + 10) as usize)])
            }
            Some(HiveMove::Move { to, .. }) => {
                Some(vec![((to.q + 10) as usize, (to.r + 10) as usize)])
            }
            _ => None,
        }
    }

    /// Check if a move is legal
    pub fn is_legal(&self, mv: &HiveMove) -> bool {
        self.get_possible_moves().contains(mv)
    }

    /// Update the cached board representation
    #[allow(dead_code)]
    fn update_cached_board(&mut self) {
        // Clear the board
        for row in &mut self.cached_board {
            for cell in row {
                *cell = 0;
            }
        }
        
        // Map hex coordinates to grid, centered at (10, 10)
        for (coord, stack) in &self.board {
            if let Some(top) = stack.last() {
                let grid_x = (coord.q + 10) as usize;
                let grid_y = (coord.r + 10) as usize;
                if grid_x < 21 && grid_y < 21 {
                    self.cached_board[grid_y][grid_x] = top.player;
                }
            }
        }
        
        self.cached_board_valid = true;
    }

    /// Get all pieces of a player currently on the board
    pub fn get_player_pieces_on_board(&self, player: i32) -> Vec<(HexCoord, &Piece)> {
        let mut pieces = Vec::new();
        for (coord, stack) in &self.board {
            if let Some(top) = stack.last() {
                if top.player == player {
                    pieces.push((*coord, top));
                }
            }
        }
        pieces
    }

    /// Get the board as hex coordinates for rendering
    pub fn get_hex_board(&self) -> &HashMap<HexCoord, Vec<Piece>> {
        &self.board
    }

    /// Get current turn number
    pub fn get_turn(&self) -> usize {
        self.turn
    }
}

impl Default for HiveState {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for HiveState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Turn {}, Player {}", self.turn, if self.current_player == 1 { "1" } else { "2" })?;
        
        // Find bounds
        let mut min_q = 0;
        let mut max_q = 0;
        let mut min_r = 0;
        let mut max_r = 0;
        
        for coord in self.board.keys() {
            min_q = min_q.min(coord.q);
            max_q = max_q.max(coord.q);
            min_r = min_r.min(coord.r);
            max_r = max_r.max(coord.r);
        }
        
        // Add padding
        min_q -= 1;
        max_q += 1;
        min_r -= 1;
        max_r += 1;
        
        // Print grid
        for r in min_r..=max_r {
            // Offset for hex display
            for _ in 0..((r - min_r) as usize) {
                write!(f, " ")?;
            }
            
            for q in min_q..=max_q {
                let coord = HexCoord::new(q, r);
                if let Some(top) = self.get_top_piece(&coord) {
                    let player_char = if top.player == 1 { '1' } else { '2' };
                    write!(f, "{}{} ", top.piece_type.char(), player_char)?;
                } else {
                    write!(f, " . ")?;
                }
            }
            writeln!(f)?;
        }
        
        Ok(())
    }
}

impl GameState for HiveState {
    type Move = HiveMove;

    fn get_num_players(&self) -> i32 {
        2
    }

    fn get_board(&self) -> &Vec<Vec<i32>> {
        // This is a bit of a hack - Hive doesn't fit well into a grid
        // We return a cached approximation for compatibility
        &self.cached_board
    }

    fn get_current_player(&self) -> i32 {
        self.current_player
    }

    fn get_possible_moves(&self) -> Vec<Self::Move> {
        let mut moves = Vec::new();
        let player_idx = Self::player_index(self.current_player);
        let pieces_placed = self.pieces_placed[player_idx];
        
        // Must place queen by turn 4 (4th piece placed)
        let must_place_queen = pieces_placed >= 3 && !self.queen_placed[player_idx];
        
        // Placement moves (if player has pieces in hand)
        let placement_positions = self.get_placement_positions();
        
        for piece_type in PieceType::all() {
            let count = self.pieces_in_hand(self.current_player, *piece_type);
            if count > 0 {
                // If must place queen, only queen placement is allowed
                if must_place_queen && *piece_type != PieceType::Queen {
                    continue;
                }
                
                for pos in &placement_positions {
                    moves.push(HiveMove::Place {
                        piece_type: *piece_type,
                        to: *pos,
                    });
                }
            }
        }
        
        // Movement moves (only if queen is placed)
        if self.queen_placed[player_idx] {
            for coord in self.occupied_positions().cloned().collect::<Vec<_>>() {
                if let Some(top) = self.get_top_piece(&coord) {
                    if top.player == self.current_player {
                        for dest in self.get_piece_moves(&coord) {
                            moves.push(HiveMove::Move {
                                from: coord,
                                to: dest,
                            });
                        }
                    }
                }
            }
        }
        
        // If no moves available but game isn't over, player must pass
        // This shouldn't happen often in Hive but handle it for robustness
        if moves.is_empty() && !self.is_terminal() {
            moves.push(HiveMove::Pass);
        }
        
        moves
    }

    fn make_move(&mut self, mv: &Self::Move) {
        let player_idx = Self::player_index(self.current_player);
        
        match mv {
            HiveMove::Place { piece_type, to } => {
                // Remove piece from hand
                if let Some(count) = self.hands.get_mut(&(self.current_player, *piece_type)) {
                    *count = count.saturating_sub(1);
                }
                
                // Place on board
                let piece = Piece::new(*piece_type, self.current_player);
                self.board.entry(*to).or_insert_with(Vec::new).push(piece);
                
                // Update tracking
                self.pieces_placed[player_idx] += 1;
                if *piece_type == PieceType::Queen {
                    self.queen_placed[player_idx] = true;
                }
            }
            HiveMove::Move { from, to } => {
                // Remove piece from source and add to destination
                // Using a two-step process to avoid borrow conflicts
                let piece = self.board.get_mut(from).and_then(|stack| stack.pop());
                
                if let Some(p) = piece {
                    // Clean up empty source stack
                    if self.board.get(from).map(|s| s.is_empty()).unwrap_or(false) {
                        self.board.remove(from);
                    }
                    // Add to destination
                    self.board.entry(*to).or_insert_with(Vec::new).push(p);
                }
            }
            HiveMove::Pass => {
                // Do nothing
            }
        }
        
        self.last_move = Some(mv.clone());
        self.current_player = -self.current_player;
        self.turn += 1;
        self.cached_board_valid = false;
    }

    fn is_terminal(&self) -> bool {
        self.get_winner().is_some()
    }

    fn get_winner(&self) -> Option<i32> {
        let p1_surrounded = self.is_queen_surrounded(1);
        let p2_surrounded = self.is_queen_surrounded(-1);
        
        match (p1_surrounded, p2_surrounded) {
            (true, true) => None,   // Draw (both surrounded on same turn)
            (true, false) => Some(-1), // Player 2 wins
            (false, true) => Some(1),  // Player 1 wins
            (false, false) => None,    // Game continues
        }
    }

    fn get_last_move(&self) -> Option<Vec<(usize, usize)>> {
        self.get_last_move()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_game() {
        let state = HiveState::new();
        assert_eq!(state.current_player, 1);
        assert_eq!(state.turn, 1);
        assert!(!state.is_terminal());
    }

    #[test]
    fn test_first_move() {
        let state = HiveState::new();
        let moves = state.get_possible_moves();
        // First player can place any of their 5 piece types at origin
        assert!(moves.len() >= 5);
    }

    #[test]
    fn test_hex_neighbors() {
        let center = HexCoord::new(0, 0);
        let neighbors = center.neighbors();
        assert_eq!(neighbors.len(), 6);
    }
}
