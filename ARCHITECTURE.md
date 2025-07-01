# Parallel MCTS Arena Architecture

This document describes the architecture of the Parallel Multi-Game MCTS Arena, a Rust terminal application that provides multiple board games with AI opponents powered by Monte Carlo Tree Search.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PARALLEL MCTS ARENA                                   │
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐  │
│  │   Terminal UI   │    │   Application   │    │    AI Engine (MCTS)         │  │
│  │   (TUI Layer)   │◄──►│   State & Logic │◄──►│   Parallel Search Tree      │  │
│  │                 │    │                 │    │                             │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────────┘  │
│           │                       │                          │                  │
│           │                       │                          │                  │
│           ▼                       ▼                          ▼                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐  │
│  │   Input System  │    │   Game Factory  │    │    Thread Pool Manager      │  │
│  │ (Mouse/Keyboard)│    │   & Wrapper     │    │  (Rayon + Custom Workers)   │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────────┘  │
│                                   │                                             │
│                                   ▼                                             │
│                          ┌──────────────────┐                                   │
│                          │   Game Engines   │                                   │
│                          │ (Othello, Gomoku,│                                   │
│                          │ Connect4, Blokus)│                                   │
│                          └──────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Application Layer (`src/main.rs` & `src/app.rs`)

The main application orchestrates all components and maintains global state:

```rust
pub struct App {
    // Core State
    pub should_quit: bool,
    pub mode: AppMode,
    pub game_wrapper: GameWrapper,
    pub game_status: GameStatus,
    
    // UI State
    pub board_cursor: (u16, u16),
    pub layout_config: LayoutConfig,
    pub drag_state: DragState,
    
    // Game Management
    pub games: Vec<(&'static str, Box<dyn Fn() -> GameWrapper>)>,
    pub player_options: Vec<(i32, Player)>,
    pub move_history: Vec<MoveHistoryEntry>,
    
    // AI Integration
    pub ai_worker: AIWorker,
    pub last_search_stats: Option<SearchStatistics>,
    
    // Settings & Configuration
    pub timeout_secs: u64,
    pub ai_minimum_display_duration: Duration,
    // ... other configuration fields
}
```

**Key Responsibilities:**
- Application state management
- Game lifecycle coordination
- UI mode transitions
- Player configuration
- AI/Human player coordination

### 2. Game Engine Layer (`src/games/` & `src/game_wrapper.rs`)

#### Game Implementations
Each game implements the `GameState` trait from the MCTS library:

```
src/games/
├── mod.rs           # Game module exports
├── othello.rs       # Othello/Reversi game logic
├── connect4.rs      # Connect Four game logic
├── gomoku.rs        # Gomoku/Five-in-a-row game logic
└── blokus.rs        # Blokus polyomino game logic
```

#### Game Wrapper System
The `GameWrapper` enum provides a unified interface for all games:

```rust
pub enum GameWrapper {
    Gomoku(GomokuState),
    Connect4(Connect4State),
    Othello(OthelloState),
    Blokus(BlokusState),
}

pub enum MoveWrapper {
    Gomoku(GomokuMove),
    Connect4(Connect4Move),
    Othello(OthelloMove),
    Blokus(BlokusMove),
}
```

**Key Features:**
- Unified game interface
- Move validation and execution
- State serialization/deserialization
- Win condition checking

### 3. AI Engine (`src/lib.rs` - MCTS Implementation)

The AI system uses a parallel Monte Carlo Tree Search implementation:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AI WORKER ARCHITECTURE                         │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │  UI Thread      │    │  AI Worker      │    │   MCTS Thread Pool      │  │
│  │                 │    │   Thread        │    │                         │  │
│  │  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌─────┐ ┌─────┐ ┌───┐  │  │
│  │  │ AIRequest │  │───►│  │ Search    │  │───►│  │ T1  │ │ T2  │ │...│  │  │
│  │  └───────────┘  │    │  │ Manager   │  │    │  └─────┘ └─────┘ └───┘  │  │
│  │                 │    │  └───────────┘  │    │                         │  │
│  │  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌─────────────────────┐│  │
│  │  │AIResponse │  │◄───│  │ Stats     │  │◄───│  │   Shared Tree       ││  │
│  │  └───────────┘  │    │  │ Collector │  │    │  │   (RwLock)          ││  │
│  └─────────────────┘    │  └───────────┘  │    │  └─────────────────────┘│  │
│                         └─────────────────┘    └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

**MCTS Features:**
- **Parallel Search**: Multi-threaded tree exploration using Rayon
- **Virtual Losses**: Prevents thread contention during search
- **PUCT Selection**: Advanced UCB1 with prior probabilities
- **Tree Reuse**: Maintains search tree between moves
- **Dynamic Allocation**: Node recycling and memory management

### 4. Terminal UI Layer (`src/tui/`)

The TUI system provides a rich terminal interface using Ratatui:

```
src/tui/
├── mod.rs           # TUI module coordination & event loop
├── widgets.rs       # UI components & rendering
├── input.rs         # Keyboard input handling
├── mouse.rs         # Mouse interaction system
├── layout.rs        # Dynamic layout management
└── blokus_ui.rs     # Blokus-specific UI components
```

#### UI Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              UI RENDERING PIPELINE                          │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │   Input Events  │    │   App State     │    │   Widget Rendering      │  │
│  │                 │    │                 │    │                         │  │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────────────┐ │  │
│  │ │  Keyboard   │ │───►│ │   Update    │ │───►│ │  Game Selection     │ │  │
│  │ └─────────────┘ │    │ │   State     │ │    │ └─────────────────────┘ │  │
│  │ ┌─────────────┐ │    │ └─────────────┘ │    │ ┌─────────────────────┐ │  │
│  │ │    Mouse    │ │    │ ┌─────────────┐ │    │ │  Settings Menu      │ │  │
│  │ └─────────────┘ │    │ │   Layout    │ │    │ └─────────────────────┘ │  │
│  │ ┌─────────────┐ │    │ │   Config    │ │    │ ┌─────────────────────┐ │  │
│  │ │   Resize    │ │    │ └─────────────┘ │    │ │  Player Config      │ │  │
│  │ └─────────────┘ │    │ ┌─────────────┐ │    │ └─────────────────────┘ │  │
│  └─────────────────┘    │ │   Drag      │ │    │ ┌─────────────────────┐ │  │
│                         │ │   State     │ │    │ │  Game Board View    │ │  │
│                         │ └─────────────┘ │    │ └─────────────────────┘ │  │
│                         └─────────────────┘    └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

**UI Features:**
- **Multi-Mode Interface**: Game selection, settings, player config, gameplay
- **Dynamic Layouts**: Resizable panels with drag-and-drop boundaries
- **Mouse Support**: Full mouse interaction for clicks, drags, and scrolling
- **Game-Specific Views**: Specialized layouts for different game types
- **Real-time Updates**: Live AI statistics and move history

### 5. Input System (`src/tui/input.rs` & `src/tui/mouse.rs`)

The input system handles all user interactions:

```rust
// Keyboard Input Pipeline
handle_key_press() -> match app.mode {
    AppMode::GameSelection => handle_game_selection_keys(),
    AppMode::Settings => handle_settings_keys(),
    AppMode::PlayerConfig => handle_player_config_keys(),
    AppMode::InGame => handle_game_keys(),
    AppMode::GameOver => handle_game_over_keys(),
}

// Mouse Input Pipeline
handle_mouse_event() -> match kind {
    MouseEventKind::Down(MouseButton::Left) => handle_mouse_click(),
    MouseEventKind::Drag(MouseButton::Left) => handle_mouse_drag(),
    MouseEventKind::Up(MouseButton::Left) => handle_mouse_release(),
    MouseEventKind::ScrollUp/ScrollDown => handle_mouse_scroll(),
    MouseEventKind::Down(MouseButton::Right) => handle_mouse_right_click(),
}
```

## Data Flow

### 1. Application Startup
```
main() -> parse_args() -> App::new() -> tui::run()
```

### 2. Game Initialization
```
Game Selection -> start_game() -> GameWrapper::new() -> 
Player Config -> confirm_player_config() -> AppMode::InGame
```

### 3. AI Move Processing
```
AI Turn -> AIWorker::start_search() -> MCTS::search() -> 
AIResponse -> App::update() -> make_move() -> UI Update
```

### 4. Human Move Processing
```
User Input -> handle_board_click() -> validate_move() -> 
make_move() -> advance_ai_tree() -> UI Update
```

### 5. UI Rendering Loop
```
Terminal Event Loop (10 FPS):
1. app.update()           # Process AI moves, update state
2. handle_input_events()  # Process keyboard/mouse
3. widgets::render()      # Render UI based on app.mode
4. terminal.draw()        # Push to terminal
```

## Key Design Patterns

### 1. State Machine Pattern
The application uses a clear state machine for UI modes:
```rust
pub enum AppMode {
    GameSelection,  // Choose game to play
    Settings,       // Configure game parameters
    PlayerConfig,   // Set human vs AI players
    InGame,         // Active gameplay
    GameOver,       // Game completed
}
```

### 2. Factory Pattern
Games are created through factory functions:
```rust
pub games: Vec<(&'static str, Box<dyn Fn() -> GameWrapper>)>
```

### 3. Wrapper Pattern
The `GameWrapper` and `MoveWrapper` enums provide unified interfaces for different game types.

### 4. Observer Pattern
The AI worker communicates with the main thread through message passing:
```rust
pub struct AIWorker {
    tx_req: Sender<AIRequest>,
    rx_resp: Receiver<AIResponse>,
}
```

### 5. Command Pattern
UI interactions are encapsulated as discrete actions:
```rust
handle_key_press() -> match key_code {
    KeyCode::Enter => confirm_selection(),
    KeyCode::Esc => go_back(),
    KeyCode::Up => move_cursor_up(),
    // ...
}
```

## Performance Considerations

### 1. Parallel MCTS
- Uses Rayon thread pool for parallel tree search
- RwLock for concurrent tree access
- Virtual losses prevent thread contention
- Configurable thread count and search limits

### 2. Memory Management
- Node recycling in MCTS tree
- Circular buffer for move history
- Lazy evaluation for UI components
- Efficient board representation

### 3. Rendering Optimization
- 10 FPS fixed update rate
- Minimal redraw on unchanged state
- Efficient terminal buffer management
- Responsive input handling

## Configuration & Extensibility

### Adding New Games
1. Implement `GameState` trait
2. Add to `games/` module
3. Update `GameWrapper` enum
4. Add factory function to `App::new()`
5. Optional: Add game-specific UI components

### AI Configuration
- Exploration constant (C_puct)
- Search iterations limit
- Node count limit
- Timeout duration
- Thread pool size

### UI Customization
- Panel size percentages
- Color schemes
- Layout configurations
- Input key mappings

This architecture provides a clean separation of concerns, making the codebase maintainable and extensible while delivering high-performance AI gameplay through parallel MCTS.
