# UI Components Documentation

This document provides detailed documentation for all UI components in the `src/components/ui/` directory. The UI system is built using a modular component architecture that promotes reusability, maintainability, and clean separation of concerns.

## Table of Contents

1. [Component Architecture Overview](#component-architecture-overview)
2. [Core UI Components](#core-ui-components)
3. [Specialized Game Components](#specialized-game-components)
4. [Layout and Theming](#layout-and-theming)
5. [Event System](#event-system)
6. [Component Lifecycle](#component-lifecycle)

## Component Architecture Overview

The UI system follows a hierarchical component model where each component is responsible for rendering a specific part of the interface and handling relevant user interactions.

### Component Hierarchy
```
RootComponent (entry point)
├── GameSelectionComponent (main menu)
├── SettingsComponent (configuration)
├── PlayerConfigComponent (player setup)
├── InGameComponent (gameplay)
│   ├── Board rendering (game-specific)
│   ├── Game info panel
│   └── Stats/history tabs
└── GameOverComponent (end game)
```

### Design Principles
- **Single Responsibility**: Each component handles one specific UI concern
- **Composability**: Components can be combined to create complex interfaces
- **Event-Driven**: Components communicate through a centralized event system
- **State Isolation**: Components only access the app state they need
- **Responsive Design**: Components adapt to different terminal sizes

## Core UI Components

### `mod.rs` - UI Module Orchestration
**File**: `src/components/ui/mod.rs`

**Purpose**: Central module definition and public interface for all UI components

**Key Exports**:
```rust
pub mod root;              // Root component and application shell
pub mod game_selection;    // Main menu and game picker
pub mod settings;          // Configuration screens
pub mod player_config;     // Player setup and AI configuration
pub mod in_game;          // Main gameplay interface
pub mod game_over;        // End game screens and results
pub mod move_history;     // Move history display and controls
pub mod board_cell;       // Individual board cell rendering
pub mod responsive_layout; // Dynamic layout management
pub mod scrollable;       // Scrollable content containers
pub mod theme;            // Color schemes and styling
```

**Responsibilities**:
- Define the public interface for the UI component system
- Ensure all components are properly exposed
- Provide centralized access to UI functionality

### `root.rs` - Application Shell
**File**: `src/components/ui/root.rs`

**Purpose**: Top-level component that orchestrates the entire application interface

**Key Structures**:
```rust
pub struct RootComponent {
    id: ComponentId,
    child_components: HashMap<ComponentId, Box<dyn Component>>,
    current_mode: AppMode,
    transition_state: TransitionState,
}
```

**Responsibilities**:
- **Mode Management**: Switch between different application screens (menu, game, settings)
- **Component Lifecycle**: Create, destroy, and manage child components
- **Global Event Handling**: Route events to appropriate child components
- **Screen Transitions**: Handle smooth transitions between different modes
- **Layout Coordination**: Manage overall application layout structure

**Key Methods**:
- `new()`: Initialize the root component with default state
- `render()`: Orchestrate rendering of the current active screen
- `handle_event()`: Route events to the appropriate child component
- `switch_mode()`: Transition between different application modes
- `get_active_component()`: Get the currently active child component

### `game_selection.rs` - Main Menu Interface
**File**: `src/components/ui/game_selection.rs`

**Purpose**: Provides the main menu interface for selecting games and accessing settings

**Key Structures**:
```rust
pub struct GameSelectionComponent {
    id: ComponentId,
    list_state: ListState,
    game_descriptions: HashMap<String, GameDescription>,
    preview_mode: bool,
}

pub struct GameDescription {
    name: String,
    description: String,
    players: String,
    complexity: Complexity,
    estimated_time: String,
}
```

**Responsibilities**:
- **Game Listing**: Display available games with descriptions and metadata
- **Navigation**: Handle menu navigation with keyboard and mouse
- **Game Preview**: Show game rules and setup information
- **Settings Access**: Provide entry point to configuration screens
- **Quick Start**: Enable rapid game launching with default settings

**UI Elements**:
- Game list with highlighting and selection
- Game description panel with rules overview
- Navigation instructions and hotkeys
- Settings and exit options

**Key Methods**:
- `render_game_list()`: Display the main game selection list
- `render_game_preview()`: Show detailed game information
- `handle_selection()`: Process game selection and launch
- `navigate_list()`: Handle keyboard/mouse navigation

### `settings.rs` - Configuration Interface
**File**: `src/components/ui/settings.rs`

**Purpose**: Comprehensive settings interface for configuring AI, display, and game parameters

**Key Structures**:
```rust
pub struct SettingsComponent {
    id: ComponentId,
    settings_categories: Vec<SettingsCategory>,
    selected_category: usize,
    selected_item: usize,
    edit_mode: bool,
    temporary_values: HashMap<String, SettingsValue>,
}

pub enum SettingsCategory {
    AI(AISettings),
    Display(DisplaySettings),
    Game(GameSettings),
    Controls(ControlSettings),
}

pub struct AISettings {
    exploration_factor: f64,
    num_threads: usize,
    search_iterations: u32,
    max_nodes: usize,
    timeout_seconds: u64,
    shared_tree: bool,
}
```

**Responsibilities**:
- **Parameter Configuration**: Allow modification of all configurable parameters
- **Input Validation**: Ensure entered values are within valid ranges
- **Real-time Preview**: Show how changes affect AI behavior or display
- **Settings Persistence**: Save and load configuration from files
- **Reset Functionality**: Restore default values when needed

**UI Sections**:
1. **AI Settings**: MCTS parameters, thread count, timeouts
2. **Display Settings**: Colors, layout preferences, animation settings
3. **Game Settings**: Board sizes, win conditions, piece sets
4. **Control Settings**: Key bindings, mouse sensitivity, shortcuts

**Key Methods**:
- `render_category_list()`: Display settings categories
- `render_settings_panel()`: Show individual settings with current values
- `handle_value_edit()`: Process settings value modifications
- `validate_settings()`: Ensure all settings are valid
- `apply_settings()`: Save changes and update application state

### `player_config.rs` - Player Setup Interface  
**File**: `src/components/ui/player_config.rs`

**Purpose**: Configure player types (human/AI) and difficulty settings for each game

**Key Structures**:
```rust
pub struct PlayerConfigComponent {
    id: ComponentId,
    player_configs: Vec<PlayerConfig>,
    selected_player: usize,
    ai_difficulty_presets: HashMap<String, AIDifficultyPreset>,
    custom_ai_settings: bool,
}

pub struct PlayerConfig {
    player_id: i32,
    player_type: Player,
    display_name: String,
    ai_settings: Option<AIPlayerSettings>,
    color_theme: ColorScheme,
}

pub struct AIDifficultyPreset {
    name: String,
    description: String,
    exploration_factor: f64,
    search_time: u64,
    strength_rating: u8,
}
```

**Responsibilities**:
- **Player Type Selection**: Choose between human and AI players
- **AI Difficulty Configuration**: Set AI strength and behavior parameters
- **Player Customization**: Configure names, colors, and preferences
- **Game-Specific Setup**: Handle different player counts for different games
- **Quick Setup**: Provide preset configurations for common scenarios

**UI Elements**:
- Player list with type indicators
- AI difficulty presets (Beginner, Intermediate, Advanced, Expert)
- Custom AI parameter controls
- Player color and name customization
- Start game button with validation

**Key Methods**:
- `render_player_list()`: Display all players with their configurations
- `render_ai_settings()`: Show AI-specific configuration options
- `handle_player_type_change()`: Switch between human and AI players
- `apply_difficulty_preset()`: Set AI parameters based on difficulty level
- `validate_configuration()`: Ensure valid setup before starting game

### `in_game.rs` - Main Gameplay Interface
**File**: `src/components/ui/in_game.rs`

**Purpose**: Primary game interface handling board display, move input, and game information

**Key Structures**:
```rust
pub struct InGameComponent {
    id: ComponentId,
    
    // Modular Blokus components (for complex Blokus UI)
    blokus_board: Option<BlokusBoardComponent>,
    blokus_piece_selector: Option<BlokusPieceSelectorComponent>,
    enhanced_blokus_piece_selector: Option<EnhancedBlokusPieceSelectorComponent>,
    improved_blokus_piece_selector: Option<ImprovedBlokusPieceSelectorComponent>,
    blokus_game_stats: Option<BlokusGameStatsComponent>,
    blokus_instruction_panel: Option<BlokusInstructionPanelComponent>,
}
```

**Responsibilities**:
- **Board Rendering**: Display current game state with appropriate styling
- **Move Input**: Handle user input for making moves (keyboard/mouse)
- **Cursor Management**: Visual cursor for move selection
- **Game Status Display**: Show current player, game phase, and status
- **AI Status Monitoring**: Display AI thinking progress and statistics
- **Move History Integration**: Show recent moves and allow history browsing
- **Game-Specific Features**: Handle unique requirements for each game type

**Layout Sections**:
1. **Game Board**: Primary game area with pieces and cursor
2. **Game Info Panel**: Current player, status, and instructions
3. **Stats/History Tabs**: AI statistics and move history with tab switching
4. **Special Panels**: Game-specific elements (piece selector for Blokus)

**Key Methods**:
- `render_game_board()`: Render the main game board for non-Blokus games
- `render_blokus_game_view()`: Specialized layout for Blokus with multiple panels
- `render_game_info()`: Display current game status and player information
- `render_stats_history_tabs()`: Tabbed interface for statistics and history
- `handle_move_input()`: Process user input for making moves
- `update_cursor_position()`: Handle cursor movement and constraints
- `make_move()`: Execute a player move and update game state

**Game-Specific Rendering**:
- **Gomoku**: Grid with X/O pieces, coordinate labels, win line highlighting
- **Connect 4**: Vertical board with piece dropping animation, column cursor
- **Othello**: 8x8 grid with disk pieces, legal move highlighting, score display
- **Blokus**: Large 20x20 grid with multi-colored pieces, piece selection panel

### `game_over.rs` - End Game Interface
**File**: `src/components/ui/game_over.rs`

**Purpose**: Display game results and provide options for continuing or returning to menu

**Key Structures**:
```rust
pub struct GameOverComponent {
    id: ComponentId,
    game_result: GameResult,
    statistics: GameStatistics,
    replay_data: Option<ReplayData>,
    celebration_animation: CelebrationState,
}

pub struct GameResult {
    winner: Option<i32>,
    final_scores: HashMap<i32, i32>,
    game_duration: Duration,
    total_moves: usize,
    ending_reason: EndingReason,
}

pub enum EndingReason {
    Victory(i32),
    Draw,
    Stalemate,
    Resignation,
    TimeExpired,
}
```

**Responsibilities**:
- **Result Display**: Show winner, scores, and game outcome
- **Statistics Summary**: Display game metrics and performance data
- **Replay Options**: Provide ability to review the completed game
- **Continue Options**: Restart, new game, or return to menu
- **Achievement Display**: Show any achievements or milestones reached

**UI Elements**:
- Large result announcement with winner celebration
- Final score breakdown by player
- Game statistics (time, moves, captures, etc.)
- Action buttons (Play Again, New Game, Main Menu)
- Move history replay controls

**Key Methods**:
- `render_result_announcement()`: Display the main game outcome
- `render_statistics_panel()`: Show detailed game metrics
- `render_action_buttons()`: Display options for continuing
- `handle_replay_controls()`: Manage move-by-move replay functionality

### `move_history.rs` - Move History Display
**File**: `src/components/ui/move_history.rs`

**Purpose**: Display and manage the chronological list of moves made during the game

**Key Structures**:
```rust
pub struct MoveHistoryComponent {
    id: ComponentId,
    display_mode: HistoryDisplayMode,
    scroll_position: usize,
    selected_move: Option<usize>,
    auto_scroll: bool,
    filter_settings: HistoryFilter,
}

pub enum HistoryDisplayMode {
    Compact,      // One line per move
    Detailed,     // Multi-line with timestamps and analysis
    Graphical,    // Visual representation of moves
}

pub struct HistoryFilter {
    show_timestamps: bool,
    show_player_names: bool,
    show_move_analysis: bool,
    player_filter: Option<i32>,
}
```

**Responsibilities**:
- **Move Display**: Show chronological list of all moves made
- **Auto-scroll**: Automatically scroll to latest moves
- **Move Selection**: Allow clicking on moves to see board state
- **Search/Filter**: Find specific moves or filter by player
- **Export**: Save move history in standard notation formats

**UI Features**:
- Scrollable list with move notation
- Timestamp display for each move
- Player indicators with color coding
- Move analysis and evaluation (when available)
- Navigation controls and keyboard shortcuts

### `board_cell.rs` - Individual Board Cell Rendering
**File**: `src/components/ui/board_cell.rs`

**Purpose**: Handle rendering and interaction for individual board cells/positions

**Key Structures**:
```rust
pub struct BoardCell {
    position: (usize, usize),
    state: CellState,
    highlight: HighlightType,
    interaction_state: InteractionState,
}

pub enum CellState {
    Empty,
    Occupied(i32),          // Player ID
    Preview(i32),           // Ghost piece preview
    Invalid,                // Invalid move position
}

pub enum HighlightType {
    None,
    Cursor,                 // Current cursor position
    LastMove,               // Recently played move
    LegalMove,              // Available move position
    WinningLine,            // Part of winning sequence
    Threat,                 // Threatens to win
}
```

**Responsibilities**:
- **Cell Rendering**: Draw individual board positions with appropriate styling
- **State Visualization**: Show piece placement, highlights, and previews
- **Interaction Handling**: Process clicks and hover events for cells
- **Animation Support**: Handle cell-level animations (piece placement, highlighting)
- **Accessibility**: Provide clear visual indicators for different states

### `responsive_layout.rs` - Dynamic Layout Management
**File**: `src/components/ui/responsive_layout.rs`

**Purpose**: Adapt UI layout to different terminal sizes and aspect ratios

**Key Structures**:
```rust
pub struct ResponsiveLayout {
    current_size: (u16, u16),
    layout_mode: LayoutMode,
    constraints: LayoutConstraints,
    cached_layouts: HashMap<(u16, u16), ComputedLayout>,
}

pub enum LayoutMode {
    Compact,      // Minimal space, stacked layout
    Standard,     // Normal layout with all panels
    Expanded,     // Extra space for detailed information
    Widescreen,   // Horizontal layout optimization
}

pub struct LayoutConstraints {
    min_board_size: (u16, u16),
    min_panel_width: u16,
    preferred_ratios: AspectRatios,
}
```

**Responsibilities**:
- **Size Detection**: Monitor terminal size changes
- **Layout Calculation**: Compute optimal layout for current size
- **Component Scaling**: Adjust component sizes proportionally
- **Layout Caching**: Cache computed layouts for performance
- **Responsive Behavior**: Switch between layout modes based on available space

### `scrollable.rs` - Scrollable Content Containers
**File**: `src/components/ui/scrollable.rs`

**Purpose**: Provide scrollable containers for content that exceeds available space

**Key Structures**:
```rust
pub struct ScrollableContainer {
    content_height: usize,
    viewport_height: usize,
    scroll_position: usize,
    scroll_behavior: ScrollBehavior,
    scrollbar_config: ScrollbarConfig,
}

pub enum ScrollBehavior {
    Manual,           // User-controlled scrolling only
    AutoScroll,       // Automatically scroll to new content
    SmartScroll,      // Auto-scroll with user override detection
}
```

**Responsibilities**:
- **Content Scrolling**: Handle vertical scrolling for long content
- **Scroll Indicators**: Show scrollbar and position indicators
- **Auto-scroll Logic**: Automatically scroll to new content when appropriate
- **User Override**: Detect when user manually scrolls and disable auto-scroll
- **Keyboard/Mouse Support**: Handle scroll wheel and keyboard navigation

### `theme.rs` - Visual Styling and Color Schemes  
**File**: `src/components/ui/theme.rs`

**Purpose**: Centralized theming system for consistent visual styling

**Key Structures**:
```rust
pub struct Theme {
    name: String,
    color_scheme: ColorScheme,
    piece_styles: HashMap<i32, PieceStyle>,
    ui_elements: UIElementStyles,
}

pub struct ColorScheme {
    background: Color,
    foreground: Color,
    accent: Color,
    player_colors: Vec<Color>,
    board_colors: BoardColors,
    ui_colors: UIColors,
}

pub struct PieceStyle {
    symbol: String,
    color: Color,
    background: Option<Color>,
    modifiers: Modifier,
}
```

**Responsibilities**:
- **Color Management**: Define consistent color schemes across the application
- **Player Styling**: Assign distinct colors and symbols to different players
- **Board Theming**: Style board elements (cells, borders, highlights)
- **UI Consistency**: Ensure consistent styling across all components
- **Accessibility**: Provide high-contrast options and colorblind-friendly themes

## Event System

### Event Flow
```
User Input (keyboard/mouse)
    ↓
Terminal Event Processing
    ↓
Component Event Translation
    ↓
Event Routing to Components
    ↓
Component Event Handling
    ↓
Application State Updates
    ↓
UI Re-rendering
```

### Event Types
- **Input Events**: Keyboard presses, mouse clicks, scroll wheel
- **Navigation Events**: Component focus changes, tab switching
- **Game Events**: Move execution, game state changes
- **System Events**: Terminal resize, application quit

## Component Lifecycle

### Lifecycle Phases
1. **Creation**: Component instantiation with initial state
2. **Initialization**: Setup of internal data structures and connections
3. **Rendering**: Continuous display updates based on current state
4. **Event Handling**: Processing user input and system events
5. **Updates**: Periodic state synchronization with application
6. **Cleanup**: Resource deallocation when component is destroyed

### State Management
- **Local State**: Component-specific data managed internally
- **Shared State**: Access to relevant parts of global application state
- **Event Communication**: Inter-component communication via event system
- **State Synchronization**: Keeping component state consistent with app state

This modular architecture ensures the UI system is maintainable, extensible, and provides a smooth user experience across all supported games and terminal environments.
