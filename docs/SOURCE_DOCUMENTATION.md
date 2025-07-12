# Source Folder Documentation

This document provides comprehensive documentation for all Rust files in the `src/` directory, explaining the purpose, structure, and key functionality of each module.

## Table of Contents

1. [Core Application Files](#core-application-files)
2. [Game Implementations](#game-implementations)
3. [Component System](#component-system)
4. [UI Components](#ui-components)
5. [Blokus-Specific Components](#blokus-specific-components)
6. [TUI Utilities](#tui-utilities)

## Core Application Files

### `main.rs` - Application Entry Point
**Purpose**: Bootstrap the application and handle command-line configuration

**Key Structures**:
- `Args`: Command-line argument parser using clap derive macros
- `main()`: Application lifecycle orchestration function

**Responsibilities**:
- Parse and validate command-line arguments
- Apply game-specific default configurations
- Initialize the main App instance with all settings
- Launch the terminal user interface event loop

**Key Features**:
- Intelligent defaults for each game type
- Thread count validation and safety checks
- Comprehensive CLI help and documentation
- Error handling for invalid configurations

---

### `lib.rs` - MCTS Engine Core
**Purpose**: Monte Carlo Tree Search algorithm implementation with parallel execution

**Key Structures**:
- `MCTS<T>`: Main search engine supporting any `GameState` implementation
- `Node<T>`: Individual nodes in the search tree with thread-safe operations
- `SearchStatistics`: Detailed analysis and metrics from search operations

**Key Features**:
- **Parallel Search**: Multi-threaded tree exploration using Rayon
- **Virtual Losses**: Prevents thread collision during parallel search
- **Tree Reuse**: Advance root functionality for efficient move transitions
- **Memory Management**: Node recycling and automatic garbage collection
- **Statistics Collection**: Comprehensive search metrics and analysis

**Thread Safety**:
- RwLock-based concurrent access to tree nodes
- Atomic counters for visit counts and values
- Lock-free algorithms where possible
- Graceful handling of thread interruption

---

### `app.rs` - Central Application State
**Purpose**: Centralized state management and AI worker coordination

**Key Structures**:
- `App`: Main application state container
- `AIWorker`: Background thread manager for AI computation
- `MoveHistoryEntry`: Individual move records for replay and analysis
- `AIRequest/AIResponse`: Message types for AI communication

**Key Enums**:
- `AppMode`: Current application screen (GameSelection, InGame, etc.)
- `GameStatus`: Current game state (InProgress, Win, Draw)
- `Player`: Player type (Human or AI)

**Responsibilities**:
- Coordinate between UI, game logic, and AI systems
- Manage application lifecycle and mode transitions
- Handle AI worker communication and status tracking
- Maintain game history and statistics
- Provide centralized configuration management

**AI Integration**:
- Non-blocking AI computation with progress tracking
- Tree sharing between moves for efficiency
- Graceful handling of AI worker failures
- Minimum display duration for smooth UX

---

### `game_wrapper.rs` - Game Abstraction Layer
**Purpose**: Unified interface for all supported game types

**Key Enums**:
- `GameWrapper`: Type-safe container for all game implementations
- `MoveWrapper`: Type-safe container for all move types

**Responsibilities**:
- Provide consistent interface across different games
- Handle move validation and execution uniformly
- Support serialization for networking (future enhancement)
- Enable generic AI algorithms to work with any game

**Design Benefits**:
- Type safety while maintaining flexibility
- Easy addition of new game types
- Centralized game logic validation
- Consistent error handling across games

## Game Implementations

### `games/mod.rs` - Game Module Coordination
**Purpose**: Module definitions and shared game interfaces

**Exports**:
- All game implementations (Gomoku, Connect4, Othello, Blokus)
- Common traits and utilities shared across games
- Game-specific move types and validation functions

---

### `games/gomoku.rs` - Five in a Row Implementation
**Purpose**: Classic Gomoku (Five in a Row) game implementation

**Key Structures**:
- `GomokuState`: Game state with configurable board size and win condition
- `GomokuMove`: Simple coordinate-based move representation

**Game Rules**:
- Variable board size (typically 15x15 or 19x19)
- Configurable win condition (typically 5 in a row)
- No captures - pure placement game
- Simple but deep strategic gameplay

**AI Considerations**:
- Moderate branching factor
- Good for testing basic MCTS functionality
- Clear win/loss conditions
- Standard opening theory available

---

### `games/connect4.rs` - Gravity-Based Four in a Row
**Purpose**: Connect 4 implementation with gravity mechanics

**Key Structures**:
- `Connect4State`: Game state with configurable dimensions
- `Connect4Move`: Column-based move (gravity determines row)

**Game Rules**:
- Typically 7 wide × 6 tall board
- Pieces fall to lowest available position
- Four in a row (horizontal, vertical, diagonal) wins
- Fast-paced tactical gameplay

**Implementation Details**:
- Efficient gravity simulation
- Quick win detection algorithms
- Optimized for tournament play
- Standard opening book integration

---

### `games/othello.rs` - Territory Control Game
**Purpose**: Othello/Reversi implementation with flanking mechanics

**Key Structures**:
- `OthelloState`: Standard 8×8 board with piece flipping logic
- `OthelloMove`: Coordinate move with automatic capture calculation

**Game Rules**:
- Fixed 8×8 board with standard starting position
- Place pieces to flank opponent pieces
- Flipped pieces change ownership
- Winner has most pieces when board is full

**Complexity Features**:
- Complex evaluation requiring territory analysis
- Sophisticated legal move generation
- Multiple capture directions per move
- Endgame analysis for exact scores

---

### `games/blokus.rs` - Multi-Player Area Control
**Purpose**: Complex 4-player territory game with polyomino pieces

**Key Structures**:
- `BlokusState`: 20×20 board with 4-player piece tracking
- `BlokusMove`: Complex move with piece ID, transformation, and position
- `BlokusPiece`: Polyomino piece definitions with transformations

**Game Rules**:
- 20×20 board with 4 players
- 21 unique polyomino pieces per player
- Pieces must touch corner-to-corner (not edge-to-edge)
- Score based on area coverage

**Implementation Complexity**:
- Large branching factor (hundreds of legal moves)
- Complex piece transformation system (rotations + reflections)
- Multi-player score evaluation
- Advanced UI requirements for piece selection

**Piece System**:
- 21 unique polyominoes from 1×1 to 5×5
- Up to 8 transformations per piece (4 rotations × 2 reflections)
- Efficient collision detection
- Legal placement validation

## Component System

### `components/mod.rs` - Component System Root
**Purpose**: Central component system exports and coordination

**Key Exports**:
- Core component traits and base functionality
- Event system for inter-component communication
- Component manager for lifecycle management
- All UI and game-specific components

---

### `components/core.rs` - Component Foundation
**Purpose**: Base traits and common functionality for all components

**Key Traits**:
- `Component`: Core interface for all UI components
- Base methods for rendering, event handling, and lifecycle management

**Key Types**:
- `ComponentId`: Unique identifier for component instances
- `ComponentResult`: Standard result type for component operations
- `EventResult`: Result type for event handling operations

**Design Philosophy**:
- Uniform interface for all UI elements
- Composable architecture for complex interfaces
- Event-driven communication between components
- Lifecycle management with proper cleanup

---

### `components/events.rs` - Event System
**Purpose**: Centralized event handling and routing system

**Key Enums**:
- `ComponentEvent`: All types of events components can receive
- `InputEvent`: User input events (keyboard, mouse)
- `SystemEvent`: Application-level events (resize, quit)

**Event Flow**:
1. Raw terminal events captured by TUI layer
2. Translated to ComponentEvent instances
3. Routed to appropriate components via component manager
4. Components process events and update application state
5. Re-rendering triggered if needed

---

### `components/manager.rs` - Component Lifecycle Management
**Purpose**: Orchestrate component creation, destruction, and event routing

**Key Structures**:
- `ComponentManager`: Central coordinator for all components
- Component tree traversal and event routing
- Focus management and tab order handling

**Responsibilities**:
- Manage component hierarchy and relationships
- Route events to focused and relevant components
- Handle component lifecycle (creation, updates, cleanup)
- Coordinate rendering order and layout

## UI Components

### `components/ui/root.rs` - Application Shell
**Purpose**: Top-level component that manages major application modes

**Key Features**:
- Mode-based component switching (GameSelection, InGame, etc.)
- Delegation pattern for clean separation of concerns
- Lifecycle management for all major UI components
- Centralized error handling and propagation

---

### `components/ui/game_selection.rs` - Main Menu Interface
**Purpose**: Game selection and main menu functionality

**Features**:
- Interactive game list with descriptions
- Game preview with rules and setup information
- Quick access to settings and configuration
- Keyboard and mouse navigation support

---

### `components/ui/settings.rs` - Configuration Interface
**Purpose**: Comprehensive settings management for all application parameters

**Settings Categories**:
- **AI Settings**: MCTS parameters, thread count, timeouts
- **Display Settings**: Colors, layout, themes
- **Game Settings**: Board sizes, win conditions
- **Control Settings**: Key bindings, mouse options

**Features**:
- Real-time parameter validation
- Preview of setting changes
- Reset to defaults functionality
- Settings persistence across sessions

---

### `components/ui/player_config.rs` - Player Setup Interface
**Purpose**: Configure player types and AI difficulty for each game

**Features**:
- Human vs AI player selection
- AI difficulty presets (Beginner to Expert)
- Custom AI parameter configuration
- Player name and color customization
- Game-specific setup validation

---

### `components/ui/in_game.rs` - Primary Gameplay Interface
**Purpose**: Main game interface for active gameplay

**Key Components**:
- **Game Board**: Adaptive rendering for all game types
- **Game Info**: Current player, status, AI progress
- **Stats/History**: Tabbed interface for analysis and move history
- **Blokus Panels**: Specialized interface for complex Blokus gameplay

**Rendering Strategies**:
- **Gomoku/Othello**: Grid-based with coordinate labels
- **Connect4**: Vertical orientation with gravity indication
- **Blokus**: Multi-panel layout with piece selector

**Input Handling**:
- Unified cursor movement across all games
- Game-specific move validation
- Mouse click-to-move support
- Keyboard shortcuts for common actions

---

### `components/ui/game_over.rs` - End Game Interface
**Purpose**: Display game results and continuation options

**Features**:
- Winner announcement with celebration effects
- Final score breakdown and statistics
- Replay functionality for move review
- Options to restart or return to menu

---

### `components/ui/move_history.rs` - Move History Display
**Purpose**: Chronological display of all moves made during the game

**Features**:
- Scrollable move list with timestamps
- Player indicators and move notation
- Click to view board state at any point
- Auto-scroll to latest moves with user override

---

### `components/ui/board_cell.rs` - Individual Board Cell Rendering
**Purpose**: Render and handle interaction for individual board positions

**Features**:
- State-specific styling (empty, occupied, highlighted)
- Click and hover handling
- Animation support for piece placement
- Accessibility features for clear state indication

---

### `components/ui/responsive_layout.rs` - Dynamic Layout Management
**Purpose**: Adapt UI layout to different terminal sizes

**Features**:
- Automatic layout mode detection
- Proportional component scaling
- Layout caching for performance
- Graceful degradation for small terminals

---

### `components/ui/scrollable.rs` - Scrollable Content Containers
**Purpose**: Handle scrolling for content that exceeds available space

**Features**:
- Vertical scrolling with scroll indicators
- Auto-scroll with user override detection
- Keyboard and mouse wheel support
- Smart scrolling behavior

---

### `components/ui/theme.rs` - Visual Styling System
**Purpose**: Centralized theming and color management

**Features**:
- Consistent color schemes across application
- Player-specific styling and symbols
- Board theming and visual effects
- Accessibility and colorblind-friendly options

## Blokus-Specific Components

### `components/blokus/board.rs` - Blokus Board Rendering
**Purpose**: Specialized rendering for the complex Blokus game board

**Features**:
- 20×20 grid with multi-colored pieces
- Ghost piece preview for move planning
- Last move highlighting
- Corner and edge connection visualization

---

### `components/blokus/piece_selector.rs` - Basic Piece Selection
**Purpose**: Simple piece selection interface for Blokus

**Features**:
- List-based piece selection
- Basic piece visualization
- Keyboard navigation
- Fallback option for resource-constrained environments

---

### `components/blokus/enhanced_piece_selector.rs` - Improved Piece Selection
**Purpose**: Enhanced piece selection with better visualization

**Features**:
- Improved piece rendering
- Better selection feedback
- More intuitive navigation
- Enhanced visual clarity

---

### `components/blokus/improved_piece_selector.rs` - Advanced Piece Selection
**Purpose**: Full-featured piece selection with all enhancements

**Features**:
- Complete piece transformation interface
- Real-time ghost preview
- Advanced keyboard shortcuts
- Mouse interaction support
- Visual transformation indicators

---

### `components/blokus/game_stats.rs` - Blokus Statistics Display
**Purpose**: Show current game statistics and player status

**Features**:
- Real-time score tracking
- Remaining piece counts
- Player ranking and status
- Game progress indicators

---

### `components/blokus/instruction_panel.rs` - Blokus Help System
**Purpose**: Contextual help and instruction display

**Features**:
- Dynamic instruction updates based on game state
- Control reference and shortcuts
- Rules clarification and examples
- Interactive help system

---

### Additional Blokus Components
- `piece_shape.rs`: Piece definition and transformation logic
- `piece_visualizer.rs`: Advanced piece rendering utilities
- `click_handler.rs`: Mouse interaction processing
- `grid_layout.rs`: Specialized grid layout management
- `enhanced_piece_grid.rs`: Advanced piece grid rendering
- `responsive_piece_grid.rs`: Adaptive piece grid layout

## TUI Utilities

### `tui/mod.rs` - TUI System Coordination
**Purpose**: Terminal user interface system exports and main event loop

**Key Functions**:
- `run()`: Main TUI event loop with terminal management
- Event processing and dispatch
- Terminal setup and cleanup
- Error handling for terminal operations

---

### `tui/layout.rs` - Layout Management System
**Purpose**: Responsive layout calculation and management

**Key Structures**:
- `LayoutConfig`: Configuration for different layout modes
- Responsive constraint calculation
- Terminal size adaptation
- Layout caching for performance

---

### `tui/mouse.rs` - Mouse Interaction Handling
**Purpose**: Comprehensive mouse event processing and state tracking

**Key Features**:
- Click position translation to game coordinates
- Drag state management for complex interactions
- Mouse wheel scrolling support
- Hover state tracking

---

### `tui/input.rs` - Input Processing
**Purpose**: Keyboard and input event handling utilities

**Features**:
- Key mapping and binding system
- Input validation and filtering
- Special key handling (arrows, function keys)
- Input mode management

---

### `tui/widgets.rs` - Custom TUI Widgets
**Purpose**: Specialized widgets for game-specific UI needs

**Custom Widgets**:
- Game board widgets for different game types
- Score displays and progress indicators
- Custom list widgets with game-specific styling
- Animation-capable widgets

---

### `tui/blokus_ui.rs` - Blokus UI Configuration
**Purpose**: Blokus-specific UI state and configuration management

**Key Structures**:
- `BlokusUIConfig`: Comprehensive Blokus UI state
- Piece selection state management
- UI mode and display preferences
- Blokus-specific interaction handling

This comprehensive documentation covers all aspects of the source code organization, providing detailed insights into the purpose, structure, and functionality of every major component in the parallel MCTS arena system.
