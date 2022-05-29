# Rust Sudoku

This project is a simple sudoku solver written in [Rust](https://www.rust-lang.org/) after completing the 
[rustlings](https://github.com/ferranSanchezLlado/rustlings) course. 

## Summary

The sudoku puzzle is a simple puzzle where the goal is to fill a 9x9 grid with digits so that each column, 
each row, and each of the nine 3x3 sub-grids that compose the grid (also called "boxes", "blocks", or "regions") 
contains all the digits from 1 to 9. The puzzle setter provides a partially completed grid, which for 
example may look like this:

```
+-----+-----+-----+
|□ 6 □|7 3 8|9 □ □|
|8 1 9|□ 2 5|6 7 3|
|5 3 □|□ □ 9|8 4 □|
+-----+-----+-----+
|9 8 3|2 □ □|4 5 6|
|4 7 2|9 5 □|□ □ □|
|1 □ 6|□ 4 3|2 9 7|
+-----+-----+-----+
|7 □ 8|5 9 1|3 6 4|
|□ □ 1|6 7 2|5 □ 9|
|6 9 □|3 8 □|7 2 1|
+-----+-----+-----+
```

The project also allows the user to change the difficulty of the puzzle and the dimensions of the grid (default is 3).
Furthermore, all methods are documented and tested.

## Installation

Basically: Clone the repository at the latest tag, run cargo install.

```bash
git clone https://github.com/ferranSanchezLlado/rust_sudoku.git
cd rust_sudoku
cargo install --force --path .
```

If there are installation errors, ensure that your toolchain is up-to-date. For the latest, run:

```bash
rustup update
```

## Future work

The project was a good introduction to Rust and probably will not be extended in the future. However, if the project 
is extended, the following features will be added:

- [ ] Implement Custom Errors and add Error Handling
- [ ] Implement a CLI where the user can play the game through the terminal
- [ ] Implement a GUI same as the CLI but through a graphical interface
