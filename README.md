# PacManAI

The following script serves as a demonstration of classical AI in a basic Pac-Man game. The game's agent, Pac-Man, is able to operate using 3 algorithms; `DFS`, `DFS_FORESEEN`, and `UCS`. Pac-Man's objective is to reach the food by as few actions as possible. However, Pac-Man only knows its own sequence of actions and positions; meaning it __can't "see" the surroundings__ unless it tries
to walk on them _(with the exception of `DFS_FORESEEN` mode which enables seeing the closest block from each side)_.

There are both a **GUI** and a **TUI** embedded into the script, which can be switched using the global variable **`gui`**.
**Step-by-step decision logs** of the agent can also be observed in the terminal by changing the global variable **`log`** to **`True`**.

![Sample preview of GUI mode](./gui_mode_preview.gif)
