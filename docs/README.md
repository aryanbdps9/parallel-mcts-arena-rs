# Parallel MCTS Arena - Complete Documentation Index

This document serves as the central index for all documentation in the parallel MCTS arena project. Each section provides links to detailed documentation about specific aspects of the codebase.

## 📚 Documentation Overview

### [🏛️ Architecture & Code Overview](CODE_DOCUMENTATION.md)
**Comprehensive overview of the entire codebase architecture**
- High-level system architecture and design principles
- Module relationships and data flow
- Thread architecture and synchronization strategy
- Key traits and interfaces
- Performance considerations and optimization strategies

### [🎮 Source Code Documentation](SOURCE_DOCUMENTATION.md)
**Detailed documentation for all source files in `/src`**
- Core application files (main.rs, app.rs, lib.rs)
- Game implementations (Gomoku, Connect4, Othello, Blokus)
- Component system architecture
- TUI utilities and terminal handling
- File-by-file breakdown with purposes and key features

### [🖥️ UI Components Documentation](UI_COMPONENTS.md)
**Complete guide to the UI component system**
- Component architecture and hierarchy
- Individual component responsibilities
- Event system and user interaction handling
- Layout management and responsive design
- Theming and visual styling system

## 🗂️ Code Organization

### Core System (`/src`)
```
src/
├── 🚀 main.rs              # Application entry point and CLI handling
├── 🧠 lib.rs               # MCTS engine core implementation
├── 📱 app.rs               # Central application state management
├── 🔄 game_wrapper.rs      # Unified game interface
├── 🎯 games/               # Individual game implementations
├── 🧩 components/          # Modular UI component system
└── 💻 tui/                 # Terminal UI utilities
```

### Game Implementations (`/src/games`)
```
games/
├── 🔴 gomoku.rs           # Five-in-a-row classic board game
├── 🟡 connect4.rs         # Gravity-based four-in-a-row
├── ⚫ othello.rs          # Territory control with piece flipping
├── 🌈 blokus.rs           # Multi-player polyomino placement
└── 📋 mod.rs              # Game module coordination
```

### Component System (`/src/components`)
```
components/
├── ⚙️  core.rs            # Base component traits and functionality
├── 📡 events.rs           # Event system and message passing
├── 👑 manager.rs          # Component lifecycle management
├── 🖼️  ui/                # Core UI components
└── 🔷 blokus/             # Blokus-specific specialized components
```

### UI Components (`/src/components/ui`)
```
ui/
├── 🏠 root.rs             # Application shell and mode switching
├── 📋 game_selection.rs   # Main menu and game picker
├── ⚙️  settings.rs        # Configuration and preferences
├── 👥 player_config.rs    # Player setup and AI configuration
├── 🎮 in_game.rs          # Primary gameplay interface
├── 🏆 game_over.rs        # End game results and options
├── 📜 move_history.rs     # Move chronology and replay
├── 🔲 board_cell.rs       # Individual cell rendering
├── 📐 responsive_layout.rs # Dynamic layout management
├── 📜 scrollable.rs       # Scrollable content containers
└── 🎨 theme.rs            # Visual styling and theming
```

### Blokus Components (`/src/components/blokus`)
```
blokus/
├── 🏁 board.rs                    # Specialized board rendering
├── 🧩 piece_selector.rs           # Basic piece selection
├── ⭐ enhanced_piece_selector.rs   # Improved piece selection
├── 🚀 improved_piece_selector.rs  # Advanced piece selection
├── 📊 game_stats.rs               # Statistics and scoring
├── 📖 instruction_panel.rs        # Help and instructions
├── 🔷 piece_shape.rs              # Piece definitions and transformations
├── 🎨 piece_visualizer.rs         # Advanced piece rendering
├── 🖱️  click_handler.rs           # Mouse interaction processing
├── 📐 grid_layout.rs              # Grid layout management
├── 🌟 enhanced_piece_grid.rs      # Advanced grid rendering
└── 📱 responsive_piece_grid.rs    # Adaptive grid layout
```

### TUI Utilities (`/src/tui`)
```
tui/
├── 🔄 mod.rs              # TUI system coordination and event loop
├── 📐 layout.rs           # Layout calculation and management
├── 🖱️  mouse.rs           # Mouse interaction and state tracking
├── ⌨️  input.rs           # Input processing and key handling
├── 🧩 widgets.rs          # Custom TUI widgets
└── 🔷 blokus_ui.rs        # Blokus-specific UI configuration
```

## 🎯 Key Features by Component

### 🧠 AI Engine (lib.rs)
- **Parallel MCTS**: Multi-threaded Monte Carlo Tree Search
- **Virtual Losses**: Prevents thread collision during search
- **Tree Reuse**: Efficient move-to-move transitions
- **Statistics**: Comprehensive search analysis and metrics
- **Memory Management**: Node recycling and garbage collection

### 📱 Application Core (app.rs)
- **State Management**: Centralized application state
- **AI Coordination**: Background thread communication
- **Mode Management**: Screen transitions and lifecycle
- **History Tracking**: Complete move history with timestamps
- **Configuration**: Runtime parameter management

### 🎮 Game Support
- **Gomoku**: Variable board size, configurable win conditions
- **Connect4**: Gravity mechanics, tournament rules
- **Othello**: Standard 8×8 with flanking captures
- **Blokus**: 4-player, 21 pieces each, complex placement rules

### 🖥️ User Interface
- **Responsive Design**: Adapts to terminal size
- **Mouse Support**: Full click and drag interactions
- **Keyboard Navigation**: Complete keyboard-only operation
- **Real-time Updates**: Live AI progress and statistics
- **Multi-game Support**: Unified interface for all games

### 🔷 Blokus Specialization
- **Piece Selection**: Advanced piece browsing and selection
- **Ghost Preview**: Real-time placement preview
- **Transformation Interface**: Piece rotation and reflection
- **Multi-panel Layout**: Complex specialized interface
- **Score Tracking**: Real-time territorial scoring

## 🛠️ Development Workflow

### Adding New Games
1. Implement `GameState` trait in `/src/games/new_game.rs`
2. Add move type to `MoveWrapper` in `game_wrapper.rs`
3. Add game variant to `GameWrapper` enum
4. Update UI components to handle new game type
5. Add game-specific rendering in `in_game.rs`

### Adding New UI Components
1. Create component file in appropriate `/src/components/` subdirectory
2. Implement `Component` trait with required methods
3. Add component to parent component's child list
4. Update event routing in component hierarchy
5. Add any required state to `App` struct

### Modifying AI Behavior
1. Adjust MCTS parameters in `lib.rs`
2. Modify exploration strategies or evaluation functions
3. Update AI difficulty presets in `player_config.rs`
4. Test with different game types for balance

## 📊 Performance Characteristics

### Memory Usage
- **MCTS Tree**: ~200 bytes per node (configurable limit)
- **Game States**: Varies by game (Blokus largest ~8KB)
- **UI Components**: Minimal overhead, mostly stack-allocated
- **Move History**: ~50 bytes per move entry

### CPU Usage
- **AI Threads**: Configurable (default 8 threads)
- **UI Thread**: Single thread for all rendering and input
- **Background Tasks**: Minimal overhead for state management

### Scalability
- **Games**: Easy to add new game types
- **AI Strength**: Scales with thread count and time allocation
- **UI Complexity**: Component system handles arbitrary complexity
- **Terminal Sizes**: Responsive design works from 80×24 to full screen

## 🚀 Quick Start Development Guide

### Understanding the Codebase
1. Start with [CODE_DOCUMENTATION.md](CODE_DOCUMENTATION.md) for architecture overview
2. Read [SOURCE_DOCUMENTATION.md](SOURCE_DOCUMENTATION.md) for file-level details
3. Review [UI_COMPONENTS.md](UI_COMPONENTS.md) for UI system understanding

### Common Development Tasks
- **Adding Features**: Identify the appropriate component and modify incrementally
- **Debugging**: Use the comprehensive logging and statistics system
- **Performance Tuning**: Focus on MCTS parameters and threading configuration
- **UI Improvements**: Leverage the component system for modular changes

### Testing Strategy
- **Unit Tests**: Each game implementation has comprehensive tests
- **Integration Tests**: Full game scenarios with AI players
- **Performance Tests**: MCTS benchmarks and memory usage validation
- **UI Tests**: Component rendering and event handling validation

This documentation system provides complete coverage of the parallel MCTS arena codebase, from high-level architecture down to individual function implementations. Each document focuses on a specific aspect while maintaining clear cross-references to related information.
