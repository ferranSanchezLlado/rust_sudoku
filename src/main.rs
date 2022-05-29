pub mod sudoku;

use sudoku::{Sudoku, Difficulty};


fn main() {
    let mut sudoku = sudoku!(&Difficulty::Easy, 3);
    println!("{}", sudoku);
    println!("{}", sudoku.is_valid());

    sudoku.reset(&Difficulty::Medium);
    println!("{}", sudoku);
    println!("{:?}", sudoku);

    sudoku.reset(&Difficulty::Hard);
    println!("{}", sudoku);
    println!("{}", sudoku.num_valid_solution());

    sudoku.solve();
    println!("{}", sudoku);
}
