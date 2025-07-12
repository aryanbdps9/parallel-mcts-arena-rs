# Blokus Component Documentation

This document provides a comprehensive overview of all the components in the Blokus UI module. The complexity stems from building a sophisticated terminal-based user interface for the Blokus board game with responsive layout, mouse interaction, and visual consistency.

## Overview

The Blokus UI system is organized into **three main categories**:

1. **Core Components** - Main game elements (board, piece selector)
2. **Supporting Components** - Player panels, statistics, instructions
3. **Utility Modules** - Shared functionality for grid layout, click handling, etc.

## File Structure & Purpose

### Core Module Files

#### `mod.rs`
**Purpose:** Module declaration and re-exports
- Declares all submodules
- Provides clean public API with re-exports
- Central import point for other modules

### Main Game Components

#### `board.rs` - Blokus Game Board
**Primary Purpose:** Renders the main 20x20 Blokus game board

**Key Structs:**
- `BlokusBoardComponent` - Main board component
- `CursorDirection` - Enum for cursor movement

**Functionality:**
- Renders the 20x20 Blokus board with colored pieces
- Handles cursor movement and positioning
- Manages board click detection and coordinate mapping
- Shows piece placement validation (legal/illegal moves)
- Displays current player's turn and game state

**Complexity Sources:**
- Terminal coordinate mapping (click → board position)
- Visual representation of different colored pieces
- Cursor positioning within borders
- Real-time move validation display

#### `improved_piece_selector.rs` - Advanced Piece Selector
**Primary Purpose:** Sophisticated piece selector with ~scrolling~, responsive layout, and advanced UX

**Key Structs:**
- `ImprovedBlokusPieceSelectorComponent` - Main selector
- `ImprovedPieceSelectorConfig` - Advanced configuration
- `ImprovedPlayerPanel` - Individual player wrapper

**Advanced Features:**
- **Scrollable Layout:** Handles overflow with scrolling
- **Current Player Priority:** Auto-expands current player's grid
- **Compact Mode:** Smaller display for non-current players
- **Dynamic Sizing:** Responsive height calculation
- **Enhanced UX:** Better visual feedback and interaction

### Piece Grid Components

#### `responsive_piece_grid.rs` - Advanced Responsive Grid
**Primary Purpose:** Sophisticated piece grid with responsive layout and precise click detection

**Key Structs:**
- `ResponsivePieceGridComponent` - Main grid component
- `ResponsivePieceGridConfig` - Comprehensive configuration

**Key Features:**
- **Responsive Layout:** Adapts grid dimensions to terminal size
- **Uniform Cell Heights:** Ensures accurate mouse click detection
- **Visual Consistency:** All pieces displayed with consistent sizing
- **Modular Design:** Uses utility modules for complex functionality

**Configuration Options:**
- `min_pieces_per_row` / `max_pieces_per_row` - Layout constraints
- `uniform_cell_height` - Critical for click detection
- `piece_width` / `piece_height` - Visual dimensions
- `show_borders` / `show_labels` - UI features
- `compact_mode` - Space-saving option
- Color theming for empty cells and pieces

**Why It's Complex:**
This is a sophisticated implementation because:
1. **Terminal UI Constraints** - Text-based rendering requires precise character positioning
2. **Mouse Interaction** - Complex coordinate mapping for click detection
3. **Dynamic Layout** - Grid adapts to screen size changes
4. **Visual Consistency** - Maintains uniform appearance across different piece shapes

#### `enhanced_piece_grid.rs` - Enhanced Grid Component
**Primary Purpose:** Clean, bordered piece grid with responsive features

**Key Structs:**
- `EnhancedPieceGridComponent` - Enhanced grid implementation  
- `EnhancedPieceGridConfig` - Configuration for enhanced features

**Features:**
- Clean border rendering similar to original design
- Responsive pieces-per-row calculation
- Simplified configuration compared to responsive grid
- Focus on visual consistency and borders

### Supporting Components

#### `player_panel.rs` - Individual Player Panel
**Primary Purpose:** Modular component for displaying one player's pieces

**Key Structs:**
- `BlokusPlayerPanelComponent` - Single player panel

**Features:**
- Expandable/collapsible interface
- Uses `EnhancedPieceGridComponent` internally
- Handles piece click events for specific player
- Dynamic height calculation

#### `piece_cell.rs` - Individual Piece Cell
**Primary Purpose:** Represents a single piece in the selector

**Key Structs:**
- `PieceCellComponent` - Single piece cell

**Features:**
- Shows piece availability (available/used)
- Handles selection state
- Click detection for piece selection
- Player-specific coloring

#### `piece_shape.rs` - Piece Shape Renderer
**Primary Purpose:** Dedicated component for rendering individual piece shapes

**Key Structs:**
- `PieceShapeComponent` - Shape renderer
- `PieceShapeConfig` - Rendering configuration

**Features:**
- Clean visual representation of piece shapes
- Configurable borders and labels
- Player color theming
- Size constraints and padding

#### `game_stats.rs` - Game Statistics Panel
**Primary Purpose:** Displays game statistics and information

**Key Structs:**
- `BlokusGameStatsComponent` - Statistics display

**Features:**
- **Multi-tab Interface:** Stats and History tabs
- **Player Statistics:** Pieces remaining, simple scoring
- **Current Player Highlighting:** Visual emphasis
- **Move History:** Track of game progression
- **Responsive Layout:** Adapts to available space

#### `instruction_panel.rs` - Help & Instructions
**Primary Purpose:** Displays game rules and controls

**Key Structs:**
- `BlokusInstructionPanelComponent` - Instruction display

**Features:**
- **Context-sensitive Help:** Different instructions for human vs AI turns
- **Control Reference:** Keyboard and mouse controls
- **Game Rules:** Blokus-specific placement rules
- **Visual Formatting:** Colored and styled text

### Utility Modules

These modules provide shared functionality used by multiple components:

#### `click_handler.rs` - Mouse Click Processing
**Primary Purpose:** Complex coordinate calculations for piece grid clicks

**Key Structs:**
- `ClickHandler` - Coordinate mapping utility

**Why It's Complex:**
Click detection in terminal UI is challenging because:
1. **Variable piece sizes** but uniform cell heights
2. **Border offsets** and grid separators
3. **Dynamic grid layouts** (pieces per row changes)
4. **Precise coordinate mapping** from mouse to grid position

**Algorithm:**
1. Account for component border offsets
2. Account for internal grid borders
3. Calculate which row was clicked (uniform cell height system)
4. Calculate which column was clicked
5. Map to piece index if valid

#### `piece_visualizer.rs` - Piece Shape Visualization
**Primary Purpose:** Converts piece coordinates to visual text representation

**Key Structs:**
- `PieceVisualizer` - Shape-to-text converter

**Challenges:**
1. **Shape normalization** - Piece coordinates can be negative or scattered
2. **Size constraints** - Must fit within fixed cell dimensions
3. **Visual clarity** - Pieces must be recognizable
4. **Consistent sizing** - All pieces use same screen space

**Algorithm:**
1. Find bounding box of piece coordinates
2. Create 2D grid large enough for the piece
3. Fill piece cells with "██" (solid blocks)
4. Convert grid to string representation
5. Apply padding for consistent sizing

#### `grid_border.rs` - Border Rendering
**Primary Purpose:** Handles drawing grid borders and separators

**Key Structs:**
- `GridBorderRenderer` - Border drawing utility

**Features:**
- Creates Unicode box-drawing characters
- Handles top/bottom borders, row separators, column separators
- Provides consistent border styling across components

#### `grid_layout.rs` - Layout Optimization
**Primary Purpose:** Calculates optimal grid arrangements

**Key Structs:**
- `GridLayoutCalculator` - Layout optimization

**Algorithm:** "Near-Square" Layout Optimization
1. Try each possible number of columns (within constraints)
2. Calculate required rows for each column count
3. Calculate aspect ratio (how "square-like" the grid is)
4. Select configuration closest to 1:1 aspect ratio

**Why This Matters:**
- Prevents extremely wide grids (21×1) that don't fit screens
- Prevents extremely tall grids (1×21) that waste horizontal space
- Creates balanced grids (~5×4) that look good and use space efficiently

## Key Design Patterns

### 1. Component Architecture
All components implement the `Component` trait with:
- `render()` - Drawing logic
- `handle_event()` - Input processing
- `id()` - Component identification

### 2. Configuration Structs
Most components use configuration structs for:
- Visual theming (colors, borders, sizing)
- Layout parameters (responsive behavior)
- Feature toggles (labels, borders, compact mode)

### 3. Modular Utilities
Complex functionality is broken into utility modules:
- Layout calculation separated from rendering
- Click handling isolated from visual logic
- Border rendering reusable across components

### 4. Responsive Design
Components adapt to terminal size using:
- Dynamic grid sizing
- Layout switching (vertical → 2×2 → horizontal)
- Minimum/maximum constraints

## Complexity Sources

### 1. Terminal UI Constraints
- Character-based positioning (no pixels)
- Unicode box-drawing for borders
- Color limitations and compatibility

### 2. Mouse Interaction
- Precise coordinate mapping
- Variable cell sizes with uniform heights
- Border and separator accounting

### 3. Responsive Layout
- Dynamic grid recalculation
- Layout switching based on available space
- Maintaining visual consistency across sizes

### 4. Game Logic Integration
- Real-time piece availability updates
- Move validation and visual feedback
- Multi-player state management

### 5. Visual Consistency
- Uniform piece sizing despite shape differences
- Consistent color theming
- Border alignment and spacing

## Architecture Summary

The Blokus component system has been streamlined to use a single, advanced piece selector implementation:

### Current Implementation
- **`improved_piece_selector.rs`** - The primary and only piece selector implementation
- **Advanced Features:** Scrolling, current player priority, responsive layout, enhanced UX
- **Simplified Architecture:** No fallback logic or redundant implementations

### Design Benefits
1. **Maintainability:** Single implementation reduces code complexity
2. **Consistency:** Unified user experience across all Blokus games
3. **Performance:** No overhead from unused fallback components
4. **Clarity:** Clear component responsibilities and interactions

The complexity is justified by the sophisticated terminal UI requirements, and the codebase now provides a clean, maintainable solution for Blokus piece selection.
