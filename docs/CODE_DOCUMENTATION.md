# Code Documentation - Parallel MCTS Arena

This document provides a comprehensive overview of the codebase structure, explaining the purpose and responsibilities of every module, struct, trait, and major function in the parallel MCTS arena project.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Source Code Structure](#source-code-structure)
3. [Core Modules](#core-modules)
4. [Component System](#component-system)
5. [Game Implementations](#game-implementations)
6. [UI System](#ui-system)
7. [Key Traits and Interfaces](#key-traits-and-interfaces)
8. [Thread Architecture](#thread-architecture)

## Architecture Overview

The parallel MCTS arena is built around several key architectural principles:

### Component-Based UI Architecture
```text
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│  Component Manager  │  Event System  │  Layout Manager     │
├─────────────────────────────────────────────────────────────┤
│     Game UI         │   Menu UI      │   Settings UI       │
├─────────────────────────────────────────────────────────────┤
│              Game Abstraction Layer                         │
├─────────────────────────────────────────────────────────────┤
│   Gomoku   │  Connect4  │  Othello   │     Blokus         │
├─────────────────────────────────────────────────────────────┤
│                    MCTS Engine                              │
└─────────────────────────────────────────────────────────────┘
```

### Thread Architecture
```text
Main Thread                AI Worker Thread           
┌─────────────┐           ┌──────────────────┐        
│             │           │                  │        
│ UI Event    │──Request─►│ MCTS Search      │        
│ Loop        │           │ Engine           │        
│             │◄Response──│                  │        
└─────────────┘           └──────────────────┘        
      │                            │                  
      ▼                            ▼                  
┌─────────────┐           ┌──────────────────┐        
│ Component   │           │ Parallel Tree    │        
│ Rendering   │           │ Search Workers   │        
└─────────────┘           └──────────────────┘        
```

## Source Code Structure

### `/src` Directory Overview

```
src/
├── main.rs              # Application entry point and CLI argument parsing
├── lib.rs               # MCTS library implementation and core algorithms  
├── app.rs               # Central application state and AI worker management
├── game_wrapper.rs      # Unified interface for all game types
├── games/               # Individual game implementations
├── components/          # Modular UI component system
└── tui/                 # Terminal user interface utilities
```

## Core Modules

### `main.rs` - Application Entry Point
**Purpose**: Application bootstrap and configuration management

**Key Components**:
- `Args` struct: Command-line argument parsing using clap
- `main()` function: Application lifecycle orchestration
- Game-specific parameter configuration
- Thread pool initialization

**Responsibilities**:
- Parse and validate command-line arguments
- Apply game-specific default configurations  
- Initialize the main App instance
- Launch the TUI event loop

### `app.rs` - Central Application State
**Purpose**: Centralized state management and coordination hub

**Key Structures**:

#### `App` Struct
The main application state container that coordinates all subsystems:

```rust
pub struct App {
    // Core application state
    pub should_quit: bool,
    pub mode: AppMode,
    pub game_wrapper: GameWrapper,
    pub game_status: GameStatus,
    
    // AI subsystem
    pub ai_worker: AIWorker,
    pub last_search_stats: Option<mcts::SearchStatistics>,
    pub pending_ai_response: Option<(MoveWrapper, mcts::SearchStatistics)>,
    
    // Game state
    pub move_history: Vec<MoveHistoryEntry>,
    pub player_options: Vec<(i32, Player)>,
    pub board_cursor: (u16, u16),
    
    // UI state
    pub layout_config: LayoutConfig,
    pub component_manager: ComponentManager,
    pub active_tab: ActiveTab,
    
    // Settings and configuration
    pub timeout_secs: u64,
    pub exploration_constant: f64,
    // ... many more fields
}
```

#### `AIWorker` Struct
Manages background AI computation:

```rust
pub struct AIWorker {
    handle: Option<JoinHandle<()>>,        // Worker thread handle
    tx_req: Sender<AIRequest>,             // Request channel
    rx_resp: Receiver<AIResponse>,         // Response channel  
    stop_flag: Arc<AtomicBool>,           // Graceful shutdown signal
}
```

**Key Methods**:
- `new()`: Creates and spawns the AI worker thread
- `start_search()`: Initiates asynchronous AI search
- `try_recv()`: Non-blocking response checking
- `advance_root()`: Updates search tree after moves
- `stop()`: Graceful worker termination

#### `MoveHistoryEntry` Struct
Tracks individual moves for replay and analysis:

```rust
pub struct MoveHistoryEntry {
    pub timestamp: SystemTime,    // When the move was made
    pub player: i32,              // Who made the move
    pub a_move: MoveWrapper,      // What move was made
}
```

### `game_wrapper.rs` - Game Abstraction Layer
**Purpose**: Provides a unified interface for all game types

**Key Components**:

#### `GameWrapper` Enum
Type-safe wrapper for all supported games:

```rust
pub enum GameWrapper {
    Gomoku(GomokuState),
    Connect4(Connect4State), 
    Othello(OthelloState),
    Blokus(BlokusState),
}
```

#### `MoveWrapper` Enum  
Type-safe wrapper for all move types:

```rust
pub enum MoveWrapper {
    Gomoku(GomokuMove),
    Connect4(Connect4Move),
    Othello(OthelloMove), 
    Blokus(BlokusMove),
}
```

**Responsibilities**:
- Provide uniform interface across different games
- Handle move validation and execution
- Manage game state transitions
- Support serialization/deserialization for networking

### `lib.rs` - MCTS Engine Implementation
**Purpose**: Core Monte Carlo Tree Search algorithm with parallel execution

**Key Structures**:

#### `MCTS<T>` Struct
The main MCTS engine supporting any game implementing `GameState`:

```rust
pub struct MCTS<T: GameState> {
    exploration_constant: f64,     // C_puct exploration parameter
    thread_pool: ThreadPool,       // Rayon thread pool for parallel search
    max_nodes: usize,             // Memory limit for search tree
    root: Arc<RwLock<Node<T>>>,   // Root node of search tree
    node_pool: Arc<Mutex<Vec<Node<T>>>>,  // Recycled nodes for efficiency
}
```

**Key Methods**:
- `new()`: Initialize MCTS engine with configuration
- `search_with_stop()`: Main search function with timeout and stop signal
- `advance_root()`: Tree reuse by advancing root after moves
- `get_statistics()`: Extract detailed search analysis

#### `Node<T>` Struct
Individual nodes in the MCTS search tree:

```rust
pub struct Node<T: GameState> {
    state: Option<T>,                    // Game position (None for virtual nodes)
    visits: AtomicI32,                   // Visit count for UCB calculation
    value: AtomicI32,                    // Total value accumulated 
    children: RwLock<Vec<Arc<RwLock<Node<T>>>>>,  // Child nodes
    parent: Option<Weak<RwLock<Node<T>>>>,        // Parent reference
    move_made: Option<T::Move>,          // Move that led to this node
    virtual_losses: AtomicI32,           // Virtual losses for parallel search
}
```

## Component System

The UI is built using a modular component system that promotes reusability and maintainability.

### Core Component Architecture

#### `Component` Trait
Base interface for all UI components:

```rust
pub trait Component {
    fn id(&self) -> ComponentId;
    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()>;
    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult;
    fn update(&mut self, app: &App) -> ComponentResult<()>;
    fn can_focus(&self) -> bool;
    fn is_focused(&self) -> bool;
    fn set_focus(&mut self, focused: bool);
}
```

#### `ComponentManager` Struct
Orchestrates component lifecycle and event routing:

```rust
pub struct ComponentManager {
    root_component: Option<Box<dyn Component>>,
    focused_component: Option<ComponentId>,
    event_queue: VecDeque<ComponentEvent>,
}
```

### Component Hierarchy

```
RootComponent
├── GameSelectionComponent
├── SettingsComponent  
├── PlayerConfigComponent
├── InGameComponent
│   ├── BoardComponent (game-specific)
│   ├── GameInfoComponent
│   └── StatsHistoryComponent
├── GameOverComponent
└── BlokusSpecificComponents/
    ├── BlokusBoardComponent
    ├── BlokusPieceSelectorComponent
    ├── BlokusGameStatsComponent
    └── BlokusInstructionPanelComponent
```

## Game Implementations

Each game implements the `GameState` trait to work with the MCTS engine.

### `GameState` Trait Requirements

```rust
pub trait GameState: Clone + Send + Sync {
    type Move: Clone + Send + Sync + Debug;
    
    fn get_current_player(&self) -> i32;
    fn get_legal_moves(&self) -> Vec<Self::Move>;
    fn make_move(&mut self, mv: &Self::Move);
    fn is_terminal(&self) -> bool;
    fn get_winner(&self) -> Option<i32>;
    fn get_reward(&self, player: i32) -> f64;
    // ... additional methods
}
```

### Game-Specific Implementations

#### Gomoku (`games/gomoku.rs`)
- **Objective**: Get 5 pieces in a row (horizontal, vertical, or diagonal)
- **Board**: Configurable NxN grid (typically 15x15 or 19x19)
- **Players**: 2 (represented as 1 and -1)
- **Complexity**: Simple rules, moderate search space

**Key Structures**:
```rust
pub struct GomokuState {
    board: Vec<Vec<i32>>,
    current_player: i32,
    board_size: usize,
    line_size: usize,
}

pub struct GomokuMove(pub usize, pub usize);  // (row, col)
```

#### Connect 4 (`games/connect4.rs`)
- **Objective**: Get 4 pieces in a row with gravity-based piece dropping
- **Board**: Typically 7 wide x 6 tall
- **Players**: 2 (represented as 1 and -1)  
- **Complexity**: Simple rules, constrained move space

**Key Structures**:
```rust
pub struct Connect4State {
    board: Vec<Vec<i32>>,
    current_player: i32,
    width: usize,
    height: usize,
    line_size: usize,
}

pub struct Connect4Move(pub usize);  // Column to drop piece
```

#### Othello/Reversi (`games/othello.rs`)
- **Objective**: Have the most pieces when the board is full
- **Board**: Fixed 8x8 grid
- **Players**: 2 (represented as 1 and -1)
- **Complexity**: Complex evaluation, flanking mechanics

**Key Structures**:
```rust
pub struct OthelloState {
    board: Vec<Vec<i32>>,
    current_player: i32,
    board_size: usize,
}

pub struct OthelloMove(pub usize, pub usize);  // (row, col)
```

#### Blokus (`games/blokus.rs`)
- **Objective**: Place as many pieces as possible while maximizing area coverage
- **Board**: Fixed 20x20 grid
- **Players**: 4 (represented as 1, 2, 3, 4)
- **Complexity**: Very high complexity, large branching factor

**Key Structures**:
```rust
pub struct BlokusState {
    board: Vec<Vec<i32>>,
    current_player: i32,
    available_pieces: HashMap<i32, HashSet<usize>>,
    last_move_coords: Option<Vec<(usize, usize)>>,
}

pub struct BlokusMove(pub usize, pub usize, pub usize, pub usize);  
// (piece_id, transformation, row, col)
```

**Blokus Pieces**: 21 unique polyominoes with up to 8 transformations each (rotations + reflections)

## UI System

### TUI Architecture (`tui/` directory)

#### Layout System (`tui/layout.rs`)
Responsive layout management for different terminal sizes:

```rust
pub struct LayoutConfig {
    pub min_board_width: u16,
    pub min_board_height: u16,
    pub preferred_info_height: u16,
    pub min_stats_width: u16,
}
```

**Key Methods**:
- `get_main_layout()`: Primary game view layout
- `get_blokus_layout()`: Specialized Blokus layout with piece selector
- `calculate_responsive_constraints()`: Dynamic sizing based on terminal dimensions

#### Mouse Support (`tui/mouse.rs`)
Comprehensive mouse interaction handling:

```rust
pub struct DragState {
    pub is_dragging: bool,
    pub start_pos: Option<(u16, u16)>,
    pub current_pos: Option<(u16, u16)>,
    pub drag_type: DragType,
}

pub enum DragType {
    None,
    BoardCursor,
    PieceSelection,
    Scrolling,
}
```

#### Blokus UI Specialization (`tui/blokus_ui.rs`)
Blokus-specific UI enhancements:

```rust
pub struct BlokusUIConfig {
    pub selected_piece_idx: Option<usize>,
    pub piece_transformation_idx: usize,
    pub piece_scroll_offset: usize,
    pub players_expanded: Vec<bool>,
    pub show_ghost_preview: bool,
}
```

### Component-Specific UI Elements

#### Board Rendering
Each game has specialized board rendering logic:
- **Gomoku/Othello**: Grid-based with coordinate labels
- **Connect4**: Column-based with gravity indication
- **Blokus**: Large grid with multi-colored pieces and ghost preview

#### Input Handling
- **Keyboard**: Arrow keys for cursor movement, Enter/Space for moves
- **Mouse**: Click to move, drag for navigation, scroll for lists
- **Game-specific**: Special keys for piece rotation (Blokus), pass moves

## Key Traits and Interfaces

### Core Game Interface
```rust
pub trait GameState: Clone + Send + Sync {
    type Move: Clone + Send + Sync + Debug;
    
    // Game state queries
    fn get_current_player(&self) -> i32;
    fn get_num_players(&self) -> i32;
    fn is_terminal(&self) -> bool;
    fn get_winner(&self) -> Option<i32>;
    
    // Move generation and validation
    fn get_legal_moves(&self) -> Vec<Self::Move>;
    fn is_legal(&self, mv: &Self::Move) -> bool;
    fn make_move(&mut self, mv: &Self::Move);
    
    // MCTS evaluation
    fn get_reward(&self, player: i32) -> f64;
    fn playout(&mut self) -> i32;
    
    // State representation
    fn get_board(&self) -> Vec<Vec<i32>>;
    fn clone_and_make_move(&self, mv: &Self::Move) -> Self;
}
```

### Component System Interfaces
```rust
pub trait Component: Send {
    fn id(&self) -> ComponentId;
    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()>;
    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult;
    fn update(&mut self, app: &App) -> ComponentResult<()>;
    fn can_focus(&self) -> bool;
    fn is_focused(&self) -> bool;
    fn set_focus(&mut self, focused: bool);
}
```

## Thread Architecture

### Main Thread Responsibilities
1. **UI Event Loop**: Process keyboard/mouse input and terminal events
2. **Component Rendering**: Update screen with current application state  
3. **AI Coordination**: Send search requests and process responses
4. **State Management**: Update game state and application mode

### AI Worker Thread Responsibilities  
1. **MCTS Search**: Perform intensive tree search computations
2. **Tree Management**: Maintain and update search tree between moves
3. **Statistics Collection**: Gather detailed analysis data
4. **Graceful Shutdown**: Respond to stop signals and cleanup resources

### Thread Communication
- **Request Channel**: Main → AI (search requests, tree updates, stop signals)
- **Response Channel**: AI → Main (best moves, search statistics)
- **Stop Signal**: Atomic boolean for immediate search interruption
- **Shared State**: None (all communication via message passing)

### Synchronization Strategy
- **Lock-Free Communication**: Message passing eliminates most synchronization needs
- **Atomic Operations**: Only for stop signals and simple counters
- **Thread-Safe MCTS**: Internal MCTS tree uses RwLocks for concurrent access
- **No Shared Mutation**: Each thread owns its data to prevent race conditions

This architecture ensures the UI remains responsive even during intensive AI computation while maintaining data consistency and preventing deadlocks.
