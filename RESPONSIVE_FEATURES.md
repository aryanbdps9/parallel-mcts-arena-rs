# Responsive TUI Features

This document describes the responsive features added to the parallel-gomoku-rs TUI.

## Features

### 1. Dynamic Window Resizing
- The TUI automatically detects terminal window size changes
- All panes adjust proportionally when the window is resized
- Layout percentages are maintained across resize operations

### 2. Draggable Pane Boundaries
The TUI now supports dragging boundaries between panes to customize the layout:

#### How to Use:
1. **Start Game**: Enter a game mode (Gomoku, Connect4, etc.)
2. **Locate Boundaries**: Look for the `↕` indicator in pane titles
3. **Drag Boundaries**: Click and drag near the border between panes
4. **Visual Feedback**: The `↕` changes to `🔀` while dragging

#### Draggable Boundaries:
- **Board ↔ Instructions**: Resize the game board vs instructions pane
- **Instructions ↔ Stats**: Resize the instructions vs debug statistics pane

### 3. Visual Indicators

#### Pane Title Indicators:
- `↕` - Shows the pane is resizable
- `🔀` - Shows dragging is active
- `(50%|25%|25%)` - Shows current layout percentages

#### Instructions:
All instruction text includes hints about the drag functionality:
- "Drag boundaries to resize panes" is added to existing instructions

### 4. Layout Constraints

#### Default Layout:
- **Board**: 50% of screen height
- **Instructions**: 20% of screen height  
- **Stats**: 30% of screen height

#### Resize Limits:
- **Board**: 30% - 80% of screen height
- **Instructions**: 5% - remaining height
- **Stats**: 5% - remaining height

### 5. State Persistence
- Layout percentages are maintained when switching between game states
- Scroll positions are reset on layout changes to prevent display issues

## Technical Implementation

### Key Components:
1. **App Fields**: Added layout tracking fields to App struct
2. **Mouse Handling**: Enhanced mouse event processing for drag operations
3. **Dynamic Layout**: Uses `app.get_layout_constraints()` instead of fixed percentages
4. **Boundary Detection**: `detect_boundary_click()` identifies draggable areas

### Mouse Events:
- **Click**: Initiates drag if clicking near boundary
- **Drag**: Updates layout percentages in real-time
- **Release**: Ends drag operation

### Layout Updates:
- Window resize updates `last_terminal_size` field
- Drag operations modify `board_height_percent`, `instructions_height_percent`, and `stats_height_percent`
- All UI components use dynamic constraints from `get_layout_constraints()`

## Usage Examples

### Maximizing Board View:
1. Drag the board-instructions boundary down to ~80%
2. Drag the instructions-stats boundary to minimize stats area

### Maximizing Debug Info:
1. Drag the instructions-stats boundary up 
2. Stats area will expand to show more debug information

### Balanced Layout:
1. Use default 50%|20%|30% for balanced gameplay and debugging
2. Adjust as needed based on terminal size and preferences

## Compatibility

- Works with all game modes (Gomoku, Connect4, Blokus, Othello)
- Compatible with existing keyboard shortcuts and AI functionality
- Maintains backward compatibility with non-mouse terminals
