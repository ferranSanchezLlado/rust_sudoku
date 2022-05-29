use rand::seq::{IteratorRandom, SliceRandom};
use std::collections::HashSet;
use std::fmt::{Display, Formatter, Result};
use strum_macros::EnumIter;

/// Creates a [`Sudoku`] with the specified [`Difficulty`] and the dimensionality of the board
/// (default is 3). The macro was mainly created to allow the dimensionality to have a default
/// value.
///
/// - Create a [`Sudoku`] with [`Difficulty::Easy`].
///
/// ```
/// let sudoku = sudoku!(&Difficulty::Easy);
/// assert_eq!(sudoku, Sudoku::new(&Difficulty::Easy, 3));
/// ```
///
/// - Create a [`Sudoku`] with [`Difficulty::Hard`] and a dimensionality of 4.
///
/// ```
/// let sudoku = sudoku!(&Difficulty::Hard, 4);
/// assert_eq!(sudoku, Sudoku::new(&Difficulty::Hard, 4));
/// ```
#[macro_export]
macro_rules! sudoku {
    ($difficulty: expr, $dimension: expr) => {
        Sudoku::new($dimension, $difficulty)
    };
    ($difficulty: expr) => {
        Sudoku::new(3, $difficulty)
    };
}

/// Specifies the maximum dimensionality that the board can have. Which corresponds to the square
/// root of the integer that is used to represent the [`Sudoku`] board.
///
/// As boards are always square, the maximum value a tile can have is two times the dimensionality
/// of the board.
///
/// Also as the board uses a non-zero-based value system, the 0 value is not allowed to be used, so
/// a the maximum dimensionality is one less than the optimal solution. However, as a computer can't
/// generate a board with such a dimensionality it doesn't matter.
pub const MAX_BOARD_DIMENSION: u16 = u8::MAX as u16;
/// Number of the tiles that are initially filled in the [`Sudoku`] board. Also, represents the
/// value of tile that can be filled during the game.
///
/// In the future, this value can be changed to `None` to use Option<u16> to represent the value of
/// a tile. However, this would require a lot of changes in the code and is not required as the
/// value of the tile is always greater than 0.
pub const EMPTY: u16 = 0;

/// Represents the difficulty of the [`Sudoku`] board. The difficulty is used to determine the
/// number tries that the [`Sudoku`] board will try to remove a random tile from the board.
///
/// The number of tries is calculated as follows:
///
/// ```text
/// Easy = pow(dimension, 2)
/// Medium = pow(dimension, 3) / 2
/// Hard = pow(dimension, 4) / 3
/// ```
#[derive(Debug, EnumIter, strum_macros::Display)]
pub enum Difficulty {
    /// Easy difficulty. The number of tries is equal to the number of tiles in a row.
    Easy,
    /// Medium difficulty. The number of tries is equal to pow(dimension, 3) / 2.
    Medium,
    /// Hard difficulty. The number of tries is equal to pow(dimension, 4) / 3.
    Hard,
}

/// A board used to represent a `Sudoku` puzzle. The board is represented by a 2D array of tiles.
/// The board is always square, and the dimensionality of the board is the square root of the
/// number of tiles in a line.
///
/// The board is first created with a [`Difficulty`] and the dimensionality of the board and
/// completely filled with a random valid solution. Then, depending on the [`Difficulty`] some
/// tiles are removed from the board.
///
///
/// # Examples
///
/// ```
/// let mut board = Sudoku::new(3, Difficulty::Easy);
///
/// println!("{}", board);
/// assert!(board.is_valid());
/// assert_eq!(board.num_valid_solutions(), 1);
///
/// board.solve();
/// board.reset(&Difficulty::Hard);
/// ```
///
/// The [`sudoku!`] macro can be used to create a [`Sudoku`] board with the specified
/// [`Difficulty`] and the dimensionality of the board. If the dimensionality is not specified,
/// the default value is 3.
///
/// ```
/// let mut board = sudoku!(&Difficulty::Easy);
/// assert!(board.is_valid());
///
/// let mut board = sudoku!(&Difficulty::Hard, 4);
/// assert!(board.is_valid());
/// ```
///
/// [`is_valid`]: Sudoku::is_valid
/// [`empty_board`]: Sudoku::empty_board
/// [`reset`]: Sudoku::reset
/// [`num_valid_solutions`]: Sudoku::num_valid_solutions
/// [`solve`]: Sudoku::solve
#[derive(Debug, Clone)]
pub struct Sudoku {
    /// Dimensionality of the board.
    dimension: u16,
    /// Board containing the sudoku currently being solved.
    board: Vec<Vec<u16>>,

    // The set of all possible values for a given cell
    /// The possibles values a row can have. Should be initialized with the range
    /// `1..=(dimension*dimension)`
    possible_row_values: Vec<HashSet<u16>>,
    /// The possibles values a column can have. Should be initialized with the range
    /// `1..=(dimension*dimension)`
    possible_col_values: Vec<HashSet<u16>>,
    /// The possibles values a box can have. Should be initialized with the range
    /// `1..=(dimension*dimension)`
    possible_box_values: Vec<HashSet<u16>>,
}

impl Sudoku {
    /// Returns the size of a line of the board. The size of a line is the square root of the
    /// the dimensionality of the board. Also, the size of a line is the square root of the
    /// number of tiles in the board.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut board = sudoku!(&Difficulty::Easy);
    /// assert_eq!(board.get_size(), 9);
    /// ```
    pub fn get_size(&self) -> u16 {
        self.dimension * self.dimension
    }

    /// Constructs a new, fully filled, [`Sudoku`] board with the specified [`Difficulty`].
    ///
    /// # Examples
    ///
    /// ```
    /// let mut board = Sudoku::new(3, Difficulty::Easy);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the dimensionality if the dimensionality is less than 2 or greater than
    /// [`MAX_BOARD_DIMENSION`].
    pub fn new(dimension: u16, difficulty: &Difficulty) -> Sudoku {
        if dimension < 2 && dimension > MAX_BOARD_DIMENSION {
            panic!("Invalid board dimension");
        }

        let size_line = (dimension * dimension) as usize;
        let mut sudoku = Sudoku {
            dimension,
            board: vec![vec![EMPTY; size_line]; size_line],
            possible_row_values: vec![HashSet::from_iter(1..=size_line as u16); size_line],
            possible_col_values: vec![HashSet::from_iter(1..=size_line as u16); size_line],
            possible_box_values: vec![HashSet::from_iter(1..=size_line as u16); size_line],
        };

        sudoku.generate_sudoku(difficulty);
        sudoku
    }

    /// Modifies the [`Sudoku`] board by setting the value of the tile at the specified row and
    /// column to the specified value. The function will manage that possible values of the
    /// tiles in the row, column and box are updated.
    ///
    /// The function will also check if the modification is valid. This is done by checking if the
    /// value is already present in the row, column and box of the tile. If the value is already
    /// present in the row, column or box, the function will return `false`. Otherwise, the function
    /// will return `true`.
    ///
    /// The checked cases are:
    ///
    /// - The previous value of the tile is not empty. In this case, the function will remove the
    ///  value from the row, column and box of the tile.
    /// - The new value is not empty. In this case, the function will add the value to the row,
    /// column and box of the tile.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut board = sudoku!(&Difficulty::Easy);
    /// board.empty_board();
    /// board.set_value(0, 0, 1);
    ///
    /// assert_eq!(board.board[0][0], 1);
    /// assert!(!board.possible_row_values[0].contains(&1));
    /// ```
    pub fn set_value(&mut self, row: u16, col: u16, value: u16) -> bool {
        let box_index = ((row / self.dimension) * self.dimension + col / self.dimension) as usize;
        let row_index = row as usize;
        let col_index = col as usize;
        let prev_value = self.board[row as usize][col as usize];
        let mut is_valid = true;

        if prev_value != EMPTY {
            is_valid &= self.possible_row_values[row_index].insert(prev_value);
            is_valid &= self.possible_col_values[col_index].insert(prev_value);
            is_valid &= self.possible_box_values[box_index].insert(prev_value);
        }
        self.board[row as usize][col as usize] = value;
        if value != EMPTY {
            is_valid &= self.possible_row_values[row_index].remove(&value);
            is_valid &= self.possible_col_values[col_index].remove(&value);
            is_valid &= self.possible_box_values[box_index].remove(&value);
        }
        is_valid
    }

    /// Gets the valid values for the tile at the specified row and column. The function will
    /// return a vector of valid values for the tile. If the tile is not empty, the function will
    /// return an empty vector.
    ///
    /// # Examples
    /// ```
    /// let mut board = sudoku!(&Difficulty::Medium);
    /// board.empty_board();
    /// board.set_value(0, 0, 1);
    /// assert_eq!(board.valid_values(0, 0), vec![2, 3, 4, 5, 6, 7, 8, 9]);
    /// ```
    fn valid_values(&self, row: u16, col: u16) -> Vec<u16> {
        if self.board[row as usize][col as usize] != EMPTY {
            return vec![];
        }

        let box_index = ((row / self.dimension) * self.dimension + col / self.dimension) as usize;
        let row_index = row as usize;
        let col_index = col as usize;
        let mut valid_values = self.possible_row_values[row_index].clone();
        valid_values.retain(|&value| {
            self.possible_col_values[col_index].contains(&value)
                && self.possible_box_values[box_index].contains(&value)
        });
        valid_values.into_iter().collect()
    }

    /// Checks if the [`Sudoku`] board respects the rules of the game. The function will return
    /// `true` if the board is valid, `false` otherwise.
    ///
    /// The function will check if the board is valid by checking if the values in each row, column
    /// and box are unique.
    ///
    /// # Examples
    /// ```
    /// let mut board = sudoku!(&Difficulty::Hard);
    /// assert!(board.is_valid());
    ///
    /// board.empty_board();
    /// assert!(board.is_valid());
    /// ```
    pub fn is_valid(&self) -> bool {
        let mut row = vec![HashSet::new(); self.get_size() as usize];
        let mut col = vec![HashSet::new(); self.get_size() as usize];
        let mut box_ = vec![HashSet::new(); self.get_size() as usize];

        for i in 0..self.get_size() {
            for j in 0..self.get_size() {
                let value = self.board[i as usize][j as usize];
                if value == EMPTY {
                    continue;
                }

                let box_index =
                    ((i / self.dimension) * self.dimension + j / self.dimension) as usize;
                let row_index = i as usize;
                let col_index = j as usize;

                if !row[row_index].insert(value)
                    || !col[col_index].insert(value)
                    || !box_[box_index].insert(value)
                {
                    return false;
                }
            }
        }
        true
    }

    /// Recursive backtracking algorithm to generate a fully filled Sudoku board that respects the
    /// rules of the game. The function will return `true` if the algorithm was able to generate a
    /// valid board, `false` otherwise.
    ///
    /// The values in the board will be randomly generated using the [`valid_values`].
    ///
    /// There function shouldn't be called directly, instead use the [`fill_board`] function
    /// which wraps the algorithm with a check to see if the board is valid.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut board = sudoku!(&Difficulty::Hard);
    /// board.empty_board();
    /// board._fill_board(0, 0);
    /// assert!(board.is_valid());
    /// ```
    ///
    /// [`valid_values`]: Sudoku::valid_values
    /// [`fill_board`]: Sudoku::fill_board
    fn _fill_board(&mut self, row: u16, col: u16) -> bool {
        if row >= self.get_size() {
            return true;
        }
        if col >= self.get_size() {
            return self._fill_board(row + 1, 0);
        }

        let mut values = self.valid_values(row, col);
        values.shuffle(&mut rand::thread_rng());
        for value in values {
            if self.set_value(row, col, value) && self._fill_board(row, col + 1) {
                return true;
            }
        }
        self.set_value(row, col, EMPTY);
        false
    }

    /// Generates a fully filled Sudoku board that respects the rules of the game. The function will
    /// return a [`Sudoku`] board that is fully filled. If the board is not valid, the function will
    /// panic.
    ///
    /// This function should be called only if the board is empty.
    ///
    /// # Examples
    /// ```
    /// let mut board = sudoku!(&Difficulty::Hard);
    /// board.empty_board();
    /// board.fill_board();
    /// assert!(board.is_valid());
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if the board is not empty or there was a problem generating a valid
    /// board.
    fn fill_board(&mut self) {
        if !self._fill_board(0, 0) {
            panic!("Could not generate a valid board, is the board empty?");
        }
    }

    /// Empties some cells in the board randomly. The function will return a [`Sudoku`] board that
    /// is fully prepared for solving. The [`Difficulty`] of the board will determine how many cells
    /// will be tried to be emptied.
    ///
    /// This function should be called only if the board is completely filled. As the function will
    /// take more time to run as more cells will be emptied, it is recommended to call this function
    /// only if the board is completely filled.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut board = sudoku!(&Difficulty::Hard);
    /// board.empty_board();
    /// board.fill_board();
    ///
    /// board.empty_some_cells(&Difficulty::Medium);
    /// assert!(board.is_valid());
    /// ```
    fn empty_some_cells(&mut self, difficulty: &Difficulty) {
        let num_steps = match difficulty {
            Difficulty::Easy => self.get_size(),
            Difficulty::Medium => self.get_size() * self.dimension / 2,
            Difficulty::Hard => self.get_size() * self.get_size() / 3,
        };
        let mut rng = rand::thread_rng();

        for _ in 0..num_steps {
            let row = (0..self.get_size()).into_iter().choose(&mut rng).unwrap() as usize;
            let col = (0..self.get_size()).into_iter().choose(&mut rng).unwrap() as usize;

            let value = self.board[row][col];
            self.set_value(row as u16, col as u16, EMPTY);
            if self.num_valid_solution() > 1 {
                self.set_value(row as u16, col as u16, value);
            }
        }
    }

    /// Generates a [`Sudoku`] board that respects the rules of the game. First, the function will
    /// generate a fully filled board. Then, the function will empty some cells in the board
    /// randomly depending on the [`Difficulty`] of the board.
    ///
    /// # Examples
    /// ```
    /// let mut board = sudoku!(&Difficulty::Hard);
    /// board.empty_board();
    ///
    /// board.generate_board(&Difficulty::Medium);
    /// assert!(board.is_valid());
    /// ```
    fn generate_sudoku(&mut self, difficulty: &Difficulty) {
        self.fill_board();
        self.empty_some_cells(difficulty);
    }

    /// Empties the board completely leaving it empty. Also restores, the internal state of the
    /// board before filling it.
    ///
    /// # Examples
    /// ```
    /// let mut board = sudoku!(&Difficulty::Hard);
    ///
    /// board.empty_board();
    /// assert!(board.is_valid());
    /// ```
    pub fn empty_board(&mut self) {
        let size = self.get_size() as usize;
        self.board = vec![vec![EMPTY; size]; size];
        self.possible_row_values = vec![HashSet::from_iter(1..=self.get_size()); size];
        self.possible_col_values = vec![HashSet::from_iter(1..=self.get_size()); size];
        self.possible_box_values = vec![HashSet::from_iter(1..=self.get_size()); size];
    }

    /// Resets the [`Sudoku`] board to a new initialized state. The board will be first emptied and
    /// then filled with a new sudoku board depending on the [`Difficulty`] of the board.
    ///
    /// # Examples
    /// ```
    /// let mut board = sudoku!(&Difficulty::Hard);
    /// board.solve(); // Solves the board
    ///
    /// board.reset(&Difficulty::Medium); // Resets the board to a new initialized state
    /// assert!(board.is_valid());
    /// ```
    pub fn reset(&mut self, difficulty: &Difficulty) {
        self.empty_board();
        self.generate_sudoku(difficulty);
    }

    /// Internal function that searches for the next empty cell in the board or until the indices
    /// are out of bounds. The functions loops first through the columns and then through the rows.
    /// The indic
    ///
    /// If the function finds an empty cell, it will return `false` and the indices will be updated
    /// with the location of the next empty cell. If the function doesn't find an empty cell and
    /// overflows the board, it will return `true`.
    ///
    /// This function is used to avoid the need to recursively call a function to find the next
    /// empty cell.
    ///
    /// # Examples
    /// ```
    /// let mut board = sudoku!(&Difficulty::Hard);
    /// board.solve(); // Solves the board
    ///
    /// // There is no empty cell
    /// assert!(board.find_next_empty_cell(&mut 0, &mut 0));
    /// ```
    fn search(&self, row: &mut u16, col: &mut u16) -> bool {
        if *row >= self.get_size()
            || *col >= self.get_size()
            || self.board[*row as usize][*col as usize] != EMPTY
        {
            loop {
                // The loop iterated through all the cells in the board, so the board is valid solution
                if *row >= self.get_size() {
                    return true;
                }
                if *col >= self.get_size() {
                    *row += 1;
                    *col = 0;
                } else if self.board[*row as usize][*col as usize] == EMPTY {
                    break;
                } else {
                    *col += 1;
                }
            }
        }
        false
    }

    /// Recursively finds the number of valid solutions for the current [`Sudoku`] board. During the
    /// execution of this function, the board will be modified. However, the board will be restored
    /// to its original state after the function returns.
    ///
    /// The function can be called publicly by using the [`num_valid_solution`] function.
    ///
    /// # Examples
    /// ```
    /// let mut board = sudoku!(&Difficulty::Hard);
    /// assert_eq!(board._num_valid_solution(0, 0), 1);
    /// ```
    ///
    /// [`num_valid_solution`]: Sudoku::num_valid_solution
    fn _num_valid_solution(&mut self, mut row: u16, mut col: u16) -> usize {
        if self.search(&mut row, &mut col) {
            return 1;
        }

        // An empty cell is found
        let mut num_valid_solution = 0;
        for value in self.valid_values(row, col) {
            if self.set_value(row, col, value) {
                num_valid_solution += self._num_valid_solution(row, col + 1);
            }
        }
        self.set_value(row, col, EMPTY);
        num_valid_solution
    }

    /// Recursively finds the number of valid solutions for the current [`Sudoku`] board. During the
    /// execution of this function, the board will be modified. However, the board will be restored
    /// to its original state after the function returns.
    ///
    /// # Examples
    /// ```
    /// let mut board = sudoku!(&Difficulty::Easy);
    /// assert_eq!(board.num_valid_solution(), 1);
    /// ```
    pub fn num_valid_solution(&mut self) -> usize {
        self._num_valid_solution(0, 0)
    }

    /// Solves the current [`Sudoku`] board. After a successful execution of this function, the
    /// board will be completely filled with valid values that satisfy the Sudoku rules. However,
    /// if the board is not valid, the board will be restored to its original state.
    ///
    /// The function can be called publicly by using the [`solve`] function.
    ///
    /// In case the board is not solvable, the function will return `false`. Otherwise, the function
    /// will return `true`.
    ///
    /// Furthermore the function is capable of solving the board even if the board has more than one
    /// solution. In this case, the function solves the first solution found.
    ///
    /// # Examples
    /// ```
    /// let mut board = sudoku!(&Difficulty::Easy);
    /// assert!(board._solve());
    /// ```
    ///
    /// [`solve`]: Sudoku::solve
    fn _solve(&mut self, mut row: u16, mut col: u16) -> bool {
        if self.search(&mut row, &mut col) {
            return true;
        }
        // An empty cell is found
        for value in self.valid_values(row, col) {
            if self.set_value(row, col, value) && self._solve(row, col + 1) {
                return true;
            }
        }
        self.set_value(row, col, EMPTY);
        false
    }

    /// Solves the current [`Sudoku`] board. After a successful execution of this function, the
    /// board will be completely filled with valid values that satisfy the Sudoku rules. However,
    /// if the board is not valid, the board will be restored to its original state.
    ///
    /// Furthermore the function is capable of solving the board even if the board has more than one
    /// solution. In this case, the function solves the first solution found.
    ///
    /// # Panics
    ///
    /// This function will panic if the board is not solvable.
    ///
    /// # Examples
    /// ```
    /// let mut board = sudoku!(&Difficulty::Easy);
    /// assert!(board._solve());
    /// ```
    pub fn solve(&mut self) {
        if !self._solve(0, 0) {
            panic!("No solution found");
        }
    }
}

impl Display for Sudoku {
    /// Displays the current [`Sudoku`] board. The board is displayed in a grid-like format.
    ///
    /// The grid is composed of a quadrant of the board, followed by a line of dashes, followed
    /// by a quadrant of the board, followed by a line of dashes, and so on.
    ///
    /// # Examples
    /// ```
    /// let mut board = sudoku!(&Difficulty::Easy);
    /// println!("{}", board);
    /// ```
    /// Example of expected output:
    /// ```text
    /// +-----+-----+-----+
    /// |□ 6 □|7 3 8|9 □ □|
    /// |8 1 9|□ 2 5|6 7 3|
    /// |5 3 □|□ □ 9|8 4 □|
    /// +-----+-----+-----+
    /// |9 8 3|2 □ □|4 5 6|
    /// |4 7 2|9 5 □|□ □ □|
    /// |1 □ 6|□ 4 3|2 9 7|
    /// +-----+-----+-----+
    /// |7 □ 8|5 9 1|3 6 4|
    /// |□ □ 1|6 7 2|5 □ 9|
    /// |6 9 □|3 8 □|7 2 1|
    /// +-----+-----+-----+
    /// ```
    ///
    fn fmt(&self, f: &mut Formatter) -> Result {
        let dimension = self.dimension as usize;
        let size_line = self.dimension * self.dimension;
        let max_length_number = self.board.iter().flatten().max().unwrap().to_string().len();
        let lines_per_quadrant = "-".repeat(dimension * max_length_number + dimension - 1);
        let quadrant_line = ("+".to_owned() + &lines_per_quadrant).repeat(dimension) + "+\n";
        for i in 0..size_line {
            if i % self.dimension == 0 {
                write!(f, "{}", quadrant_line)?;
            }
            for j in 0..size_line {
                write!(f, "{}", if j % self.dimension == 0 { "|" } else { " " })?;
                let value = self.board[i as usize][j as usize];
                let value_str = if value == 0 {
                    "□".to_string()
                } else {
                    value.to_string()
                };
                write!(f, "{:^width$}", value_str, width = max_length_number)?;
            }
            write!(f, "|\n")?;
        }
        write!(f, "{}", quadrant_line)?;
        Ok(())
    }
}

/// Module containing the tests for the [`Sudoku`] struct. The module only contains a single
/// function, [`test`], which tests the Sudoku with multiple [`Difficulty`]s and dimensions.
///
/// In the future, this module could be divided into smaller tests for each of the Sudoku
/// methods. However, this is not done at the moment.
#[cfg(test)]
mod tests {
    use strum::IntoEnumIterator;
    use crate::sudoku::{Sudoku, Difficulty};

    /// Tests that the Sudoku with multiple [`Difficulty`]s and dimensions are valid.
    /// This test is not exhaustive, but it is sufficient to ensure that the Sudoku struct is
    /// working as intended. All public methods are tested in the [`test`] function.
    ///
    /// All difficulties are tested, but only the dimensions between 2 and 4 are tested, as the
    /// Sudoku solver takes a long time to run for larger dimensions.
    #[test]
    fn test() {
        for size in 2..=4 {
            for difficulty in Difficulty::iter() {
                println!(
                    "Testing:\n - Dimension: {}\n - Difficulty: {}",
                    size, difficulty
                );
                let mut sudoku = sudoku!(&difficulty, size);
                assert!(sudoku.is_valid());
                assert_eq!(sudoku.num_valid_solution(), 1);

                // Test that there are gaps
                assert_ne!(
                    sudoku.board.iter().flatten().filter(|&&x| x == 0).count(),
                    0
                );

                // Tests solving
                sudoku.solve();
                assert!(sudoku.is_valid());
                assert_eq!(sudoku.num_valid_solution(), 1); // Maybe should return 0
                assert_eq!(
                    sudoku.board.iter().flatten().filter(|&&x| x == 0).count(),
                    0
                ); // No empty cell

                // Ensures solves obtains the original board
                sudoku.empty_board();
                sudoku.fill_board();
                let original_sudoku = sudoku.clone();
                sudoku.empty_some_cells(&difficulty);

                sudoku.solve();
                assert_eq!(sudoku.board, original_sudoku.board);
            }
        }
    }
}
