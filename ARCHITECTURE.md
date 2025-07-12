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
    
    // UI State & Layout
    pub board_cursor: (u16, u16),
    pub layout_config: LayoutConfig,
    pub drag_state: DragState,
    pub blokus_ui_config: BlokusUIConfig,
    
    // Component System
    pub component_manager: ComponentManager,
    
    // Game Management
    pub games: Vec<(&'static str, Box<dyn Fn() -> GameWrapper>)>,
    pub player_options: Vec<(i32, Player)>,
    pub move_history: Vec<MoveHistoryEntry>,
    
    // AI Integration
    pub ai_worker: AIWorker,
    pub last_search_stats: Option<SearchStatistics>,
    pub ai_thinking_start: Option<Instant>,
    pub ai_minimum_display_duration: Duration,
    pub pending_ai_response: Option<(MoveWrapper, SearchStatistics)>,
    
    // Settings & Configuration
    pub timeout_secs: u64,
    pub stats_interval_secs: u64,
    pub ai_only: bool,
    pub shared_tree: bool,
    pub settings_board_size: usize,
    pub settings_line_size: usize,
    pub settings_ai_threads: usize,
    pub settings_max_nodes: usize,
    pub settings_search_iterations: u32,
    pub settings_exploration_constant: f64,
    
    // UI Navigation State
    pub game_selection_state: ListState,
    pub settings_state: ListState,
    pub selected_player_config_index: usize,
    pub selected_settings_index: usize,
    pub active_tab: ActiveTab,
    
    // Auto-scroll & Display Features
    pub history_auto_scroll: bool,
    pub piece_panel_auto_scroll: bool,
    pub show_debug: bool,
    // ... additional UI state fields
}
```

**Key Responsibilities:**
- Application state management
- Game lifecycle coordination
- UI mode transitions via component system
- Player configuration
- AI/Human player coordination
- Component management and event routing

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
- **PUCT Selection**: Advanced UCB1 formula with prior probabilities for enhanced move selection
- **Tree Reuse**: Maintains search tree between moves for performance
- **Dynamic Allocation**: Node recycling and memory management
- **Configurable Parameters**: Exploration constant, thread count, node limits, and timeouts

### 4. Terminal UI Layer (`src/tui/` & `src/components/`)

The UI system provides a rich terminal interface using both legacy TUI utilities and a modern component system:

```
src/tui/                 # Legacy TUI utilities (being phased out)
├── mod.rs              # TUI module coordination & event loop
├── widgets.rs          # Legacy UI rendering functions
├── input.rs            # Keyboard input handling  
├── mouse.rs            # Mouse interaction system
├── layout.rs           # Dynamic layout management
└── blokus_ui.rs        # Blokus-specific UI components

src/components/          # Modern component-based UI system
├── core.rs             # Component trait definitions
├── manager.rs          # Component lifecycle management
├── events.rs           # Event system for component communication
├── ui/                 # Core UI components
│   ├── root.rs         # Application shell component
│   ├── game_selection.rs # Main menu component
│   ├── settings.rs     # Settings configuration component
│   ├── player_config.rs # Player setup component
│   ├── in_game.rs      # Main gameplay component
│   ├── game_over.rs    # End game component
│   ├── move_history.rs # Move history display
│   ├── board_cell.rs   # Individual board cell rendering
│   ├── responsive_layout.rs # Dynamic layout system
│   ├── scrollable.rs   # Scrollable content containers
│   └── theme.rs        # Visual styling and themes
└── blokus/             # Blokus-specific components
    ├── board.rs        # Blokus board rendering
    ├── piece_selector.rs # Basic piece selection
    ├── enhanced_piece_selector.rs # Enhanced piece selection
    ├── improved_piece_selector.rs # Latest piece selection UI
    ├── player_panel.rs # Individual player panels
    ├── piece_grid.rs   # Piece grid layouts
    ├── piece_shape.rs  # Individual piece shape rendering
    ├── game_stats.rs   # Game statistics display
    └── instruction_panel.rs # Instructions and help
```

#### Component-Based UI Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPONENT-BASED UI SYSTEM                           │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │   Input Events  │    │ Component       │    │   Component Rendering   │  │
│  │                 │    │ Manager         │    │                         │  │
│  │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────────────┐ │  │
│  │ │  Keyboard   │ │───►│ │   Event     │ │───►│ │  RootComponent      │ │  │
│  │ └─────────────┘ │    │ │  Routing    │ │    │ └─────────────────────┘ │  │
│  │ ┌─────────────┐ │    │ └─────────────┘ │    │ ┌─────────────────────┐ │  │
│  │ │    Mouse    │ │    │ ┌─────────────┐ │    │ │  GameSelection      │ │  │
│  │ └─────────────┘ │    │ │ Component   │ │    │ │  Settings           │ │  │
│  │ ┌─────────────┐ │    │ │ Lifecycle   │ │    │ │  PlayerConfig       │ │  │
│  │ │   Resize    │ │    │ │ Management  │ │    │ │  InGame             │ │  │
│  │ └─────────────┘ │    │ └─────────────┘ │    │ │  GameOver           │ │  │
│  └─────────────────┘    └─────────────────┘    │ └─────────────────────┘ │  │
│                                                │ ┌─────────────────────┐ │  │
│                                                │ │  Blokus Components  │ │  │
│                                                │ │  • Board            │ │  │
│                                                │ │  • PieceSelector    │ │  │
│                                                │ │  • PlayerPanels     │ │  │
│                                                │ │  • GameStats        │ │  │
│                                                │ └─────────────────────┘ │  │
│                                                └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

**UI Features:**
- **Component-Based Architecture**: Modular, reusable UI components with clear separation of concerns
- **Multi-Mode Interface**: Game selection, settings, player config, gameplay modes
- **Modern Event System**: Type-safe event routing and component communication
- **Dynamic Layouts**: Resizable panels with drag-and-drop boundaries
- **Mouse Support**: Full mouse interaction for clicks, drags, and scrolling
- **Game-Specific Views**: Specialized layouts and components for different game types
- **Real-time Updates**: Live AI statistics and move history
- **Responsive Design**: Adaptive layouts for different terminal sizes
- **Theme System**: Consistent visual styling and color schemes

### 5. Component System (`src/components/`)

The modern component system provides a structured approach to UI development:

```rust
// Core Component Trait
pub trait Component {
    fn id(&self) -> ComponentId;
    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()>;
    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

// Component Manager for lifecycle management
pub struct ComponentManager {
    components: HashMap<ComponentId, Box<dyn Component>>,
    root_component_id: Option<ComponentId>,
}

// Event system for component communication
pub enum ComponentEvent {
    Input(InputEvent),
    MouseClick(u16, u16),
    Resize(u16, u16),
    Refresh,
}
```

**Component Features:**
- **Type Safety**: Compile-time guarantees for component interactions
- **Lifecycle Management**: Automatic component creation, update, and cleanup
- **Event-Driven**: Clean separation between input handling and business logic
- **Composability**: Components can be nested and combined for complex UIs
- **Reusability**: Components can be used across different game types

### 6. Input System (`src/tui/input.rs` & `src/tui/mouse.rs`)

The input system handles all user interactions and integrates with the component system:

```rust
// Component-Integrated Input Pipeline
handle_key_press() -> ComponentEvent::Input(InputEvent::KeyPress(key)) -> 
ComponentManager::send_event() -> ActiveComponent::handle_event()

// Legacy Direct Input Pipeline (being phased out)
handle_key_press() -> match app.mode {
    AppMode::GameSelection => handle_game_selection_keys(),
    AppMode::Settings => handle_settings_keys(),
    AppMode::PlayerConfig => handle_player_config_keys(),
    AppMode::InGame => handle_game_keys(),
    AppMode::GameOver => handle_game_over_keys(),
}

// Component-Integrated Mouse Pipeline
handle_mouse_event() -> ComponentEvent::MouseClick(x, y) ->
ComponentManager::send_event() -> ActiveComponent::handle_event()

// Legacy Direct Mouse Pipeline (being phased out)
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
1. app.update()                    # Process AI moves, update state
2. handle_input_events()           # Process keyboard/mouse input
3. component_manager.render()      # Render active components (preferred)
   OR widgets::render()            # Legacy rendering (being phased out)
4. terminal.draw()                 # Push to terminal
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

### 6. Component Pattern
The modern UI system uses a component-based architecture:
```rust
pub trait Component {
    fn render(&mut self, frame: &mut Frame, area: Rect, app: &App) -> ComponentResult<()>;
    fn handle_event(&mut self, event: &ComponentEvent, app: &mut App) -> EventResult;
}
```

Components are managed by the ComponentManager and communicate through events.
### 7. Command Pattern
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
5. Optional: Add game-specific UI components to `components/` directory

### AI Configuration
- Exploration constant (C_puct)
- Search iterations limit
- Node count limit
- Timeout duration
- Thread pool size

### UI Customization
- Component configuration and themes
- Panel size percentages and layout constraints
- Color schemes and visual styling
- Layout configurations and responsive design
- Input key mappings and event bindings
- Animation and transition settings

This architecture provides a clean separation of concerns with a modern component-based UI system, making the codebase maintainable and extensible while delivering high-performance AI gameplay through parallel MCTS.
