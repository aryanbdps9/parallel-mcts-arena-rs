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
- **Board**: Minimum height based on board size + some extra space
- **Instructions**: Minimum 3 lines + some extra space
- **Stats**: Remaining space (minimum 5% of screen)

#### Resize Limits:
- **Board**: Cannot be reduced below minimum height needed for game board
- **Instructions**: Cannot be reduced below minimum height needed for text display  
- **Stats**: Cannot be reduced below 5% of screen height (scrollable content)

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

### Initial Layout:
- The application automatically calculates minimum heights based on content
- Board height is set to accommodate the full game board plus borders
- Instructions height is set to the minimum needed for text display
- Stats area gets the remaining space for debug information

### Maximizing Board View:
1. Drag the board-instructions boundary down to expand board area
2. Board cannot be reduced below the minimum size needed for gameplay
3. Drag the instructions-stats boundary to minimize stats area

### Maximizing Debug Info:
1. Drag the instructions-stats boundary up 
2. Stats area will expand to show more debug information
3. Instructions section cannot be reduced below minimum readable height

### Adaptive Behavior:
- Switching between games automatically adjusts minimum heights
- Gomoku 15x15 needs more space than Connect4 7x6
- Window resizing respects content requirements and maintains usability

## Compatibility

- Works with all game modes (Gomoku, Connect4, Blokus, Othello)
- Compatible with existing keyboard shortcuts and AI functionality
- Maintains backward compatibility with non-mouse terminals
