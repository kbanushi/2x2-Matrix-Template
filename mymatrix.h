///@author Krenar Banushi
///@date Feburary 28, 2023
///@brief This is a header file to implement a templated 2x2 matrix.  The memory is dynamically allocated in the heap and supports growing, copying, and scalar and matrix multiplication.
/// Assignment details and provided code are created and
/// owned by Adam T Koehler, PhD - Copyright 2023.
/// University of Illinois Chicago - CS 251 Spring 2023


//
// mymatrix
//
// The mymatrix class provides a matrix (2D array) abstraction.
// The size can grow dynamically in both directions (rows and 
// cols).  Also, rows can be "jagged" --- i.e. rows can have 
// different column sizes, and thus the matrix is not necessarily 
// rectangular.  All elements are initialized to the default value
// for the given type T.  Example:
//
//   mymatrix<int>  M;  // 4x4 matrix, initialized to 0
//   
//   M(0, 0) = 123;
//   M(1, 1) = 456;
//   M(2, 2) = 789;
//   M(3, 3) = -99;
//
//   M.growcols(1, 8);  // increase # of cols in row 1 to 8
//
//   for (int r = 0; r < M.numrows(); ++r)
//   {
//      for (int c = 0; c < M.numcols(r); ++c)
//         cout << M(r, c) << " ";
//      cout << endl;
//   }
//
// Output:
//   123 0 0 0
//   0 456 0 0 0 0 0 0
//   0 0 789 0
//   0 0 0 -99
//

#pragma once

#include <iostream>
#include <exception>
#include <stdexcept>

using namespace std;

template<typename T>
class mymatrix
{
private:
  struct ROW
  {
    T*  Cols;     // dynamic array of column elements
    int NumCols;  // total # of columns (0..NumCols-1)
  };

  ROW* Rows;     // dynamic array of ROWs
  int  NumRows;  // total # of rows (0..NumRows-1)
  int NumElements;
  
  /// @brief Increases the number of elements in a column using the index of which row to increase and the new size.  Does not free old memory.
  /// @param index The index of which row we are changing the size of 
  /// @param newSize The new size
  void IncreaseColumnElements(int index, int newSize){
    T* newArray = new T[newSize];
    for (int i = 0; i < newSize; i++){
      if (i < Rows[index].NumCols)
        newArray[i] = Rows[index].Cols[i];
      else
        newArray[i] = T{};
    }
    Rows[index].Cols = newArray;
    NumElements += (newSize - Rows[index].NumCols);
    Rows[index].NumCols = newSize;
  }

  /// @brief Increases the number of rows to the new size and each new row will be of size colSize
  /// @param newSize The number of rows to update the ROW array to
  /// @param colSize How many columns each new row should contain
  void IncreaseRowElements(int newSize, int colSize){
    ROW* newRows = new ROW[newSize];

    for (int row = 0; row < newSize; row++){
      if (row < NumRows){
        newRows[row].Cols = Rows[row].Cols;
        newRows[row].NumCols = Rows[row].NumCols;
      }
      else{
        newRows[row].Cols = new T[colSize];
        newRows[row].NumCols = colSize;
        for (int c = 0; c < newRows[row].NumCols; c++){
          newRows[row].Cols[c] = T{};
        }
      }
    }
    NumElements += ((newSize - NumRows) * colSize);
    this->Rows = newRows;
    this->NumRows = newSize;
  }

  /// @brief Using a singular row from the first matrix and all of the rows from the second matrix, return result after matrix multiplication
  /// @param M1Row Singular row from first matrix
  /// @param M2Rows Array of rows from second matrix
  /// @param pos Column position to access in second matrix
  /// @return Result of matrix multiplication from first matrix row and second matrix column
  T CalcMatrixElement(ROW M1Row, ROW* M2Rows, int pos){
    T result = 0;
    int numPos = M1Row.NumCols;
    for (int i = 0; i < numPos; i++){
      result += (M1Row.Cols[i] * M2Rows[i].Cols[pos]);
    }

    return result;
  }
  
  /// @brief Check if a given matrix is rectangular (ie all rows contain the same number of columns)
  /// @return True if the given matrix is rectangular by definition or false otherwise
  bool IsRectangular() const {
    int firstColElements = Rows[0].NumCols;

    for (int i = 1; i < NumRows; i++){
      if (Rows[i].NumCols != firstColElements)
        return false;
    }
    return true;
  }

public:
  //
  // default constructor:
  //
  // Called automatically by C++ to construct a 4x4 matrix.  All 
  // elements are initialized to the default value of T.
  //
  mymatrix()
  {
    Rows = new ROW[4];  // an array with 4 ROW structs:
    NumRows = 4;
    NumElements = 16;

    // initialize each row to have 4 columns:
    for (int r = 0; r < NumRows; ++r)
    {
      Rows[r].Cols = new T[4];  // an array with 4 elements of type T:
      Rows[r].NumCols = 4;

      // initialize the elements to their default value:
      for (int c = 0; c < Rows[r].NumCols; ++c)
      {
        Rows[r].Cols[c] = T{};  // default value for type T:
      }
    }
  }

  //
  // parameterized constructor:
  //
  // Called automatically by C++ to construct a matrix with R rows, 
  // where each row has C columns. All elements are initialized to 
  // the default value of T.
  //
  mymatrix(int R, int C)
  {
    if (R < 1)
      throw invalid_argument("mymatrix::constructor: # of rows");
    if (C < 1)
      throw invalid_argument("mymatrix::constructor: # of cols");

    NumRows = R;
    Rows = new ROW[NumRows]; //Array with NumRows ROW structs
    NumElements = R * C;

    for (int r = 0; r < NumRows; r++){ 
      Rows[r].Cols = new T[C]; //Array of type T with C number of elements
      Rows[r].NumCols = C;

      for (int col = 0; col < C; col++){
        Rows[r].Cols[col] = T{};
      }
    }
  }


  //
  // copy constructor:
  //
  // Called automatically by C++ to construct a matrix that contains a 
  // copy of an existing matrix.  Example: this occurs when passing 
  // mymatrix as a parameter by value
  //
  //   void somefunction(mymatrix<int> M2)  <--- M2 is a copy:
  //   { ... }
  //
  mymatrix(const mymatrix<T>& other)
  {
    NumRows = other.NumRows;
    Rows = new ROW[NumRows]; //Array with NumRows ROW structs
    this->NumElements = other.NumElements; 

    //Deep copy values from other matrix
    for (int r = 0; r < NumRows; r++){
      int otherNumCols = other.Rows[r].NumCols;
      Rows[r].Cols = new T[otherNumCols];
      Rows[r].NumCols = otherNumCols;

      for (int col = 0; col < otherNumCols; col++){
        Rows[r].Cols[col] = other.Rows[r].Cols[col];
      }
    }
  }


  //
  // numrows
  //
  // Returns the # of rows in the matrix.  The indices for these rows
  // are 0..numrows-1.
  //
  int numrows() const
  {
    return NumRows;
  }
  

  //
  // numcols
  //
  // Returns the # of columns in row r.  The indices for these columns
  // are 0..numcols-1.  Note that the # of columns can be different 
  // row-by-row since "jagged" rows are supported --- matrices are not
  // necessarily rectangular.
  //
  int numcols(int r) const
  {
    if (r < 0 || r >= NumRows)
      throw invalid_argument("mymatrix::numcols: row");

    return Rows[r].NumCols;
  }


  //
  // growcols
  //
  // Grows the # of columns in row r to at least C.  If row r contains 
  // fewer than C columns, then columns are added; the existing elements
  // are retained and new locations are initialized to the default value 
  // for T.  If row r has C or more columns, then all existing columns
  // are retained -- we never reduce the # of columns.
  //
  // Jagged rows are supported, i.e. different rows may have different
  // column capacities -- matrices are not necessarily rectangular.
  //
  void growcols(int r, int C)
  {
    if (r < 0 || r >= NumRows)
      throw invalid_argument("mymatrix::growcols: row");
    if (C < 1)
      throw invalid_argument("mymatrix::growcols: columns");

    if (Rows[r].NumCols < C)
      IncreaseColumnElements(r, C);
  }


  //
  // grow
  //
  // Grows the size of the matrix so that it contains at least R rows,
  // and every row contains at least C columns.
  // 
  // If the matrix contains fewer than R rows, then rows are added
  // to the matrix; each new row will have C columns initialized to 
  // the default value of T.  If R <= numrows(), then all existing
  // rows are retained -- we never reduce the # of rows.
  //
  // If any row contains fewer than C columns, then columns are added
  // to increase the # of columns to C; existing values are retained
  // and additional columns are initialized to the default value of T.
  // If C <= numcols(r) for any row r, then all existing columns are
  // retained -- we never reduce the # of columns.
  // 
  void grow(int R, int C)
  {
    if (R < 1)
      throw invalid_argument("mymatrix::grow: # of rows");
    if (C < 1)
      throw invalid_argument("mymatrix::grow: # of cols");

    if (NumRows < R)
      IncreaseRowElements(R, C);
    
    for (int row = 0; row < NumRows; row++){
      if (Rows[row].NumCols < C)
        IncreaseColumnElements(row, C);
    }
  }


  //
  // size
  //
  // Returns the total # of elements in the matrix.
  //
  int size() const
  {
    return NumElements;
  }


  //
  // at
  //
  // Returns a reference to the element at location (r, c); this
  // allows you to access the element or change it:
  //
  //    M.at(r, c) = ...
  //    cout << M.at(r, c) << endl;
  //
  T& at(int r, int c) const
  {
    if (r < 0 || r >= NumRows)
      throw invalid_argument("mymatrix::at: row");
    if (c < 0 || c >= Rows[r].NumCols)
      throw invalid_argument("mymatrix::at: col");

    return Rows[r].Cols[c];
  }


  //
  // ()
  //
  // Returns a reference to the element at location (r, c); this
  // allows you to access the element or change it:
  //
  //    M(r, c) = ...
  //    cout << M(r, c) << endl;
  //
  T& operator()(int r, int c) const
  {
    if (r < 0 || r >= NumRows)
      throw invalid_argument("mymatrix::(): row");
    if (c < 0 || c >= Rows[r].NumCols)
      throw invalid_argument("mymatrix::(): col");

    return Rows[r].Cols[c];

  }
  //
  // scalar multiplication
  //
  // Multiplies every element of this matrix by the given scalar value,
  // producing a new matrix that is returned.  "This" matrix is not
  // changed.
  //
  // Example:  M2 = M1 * 2;
  //
  mymatrix<T> operator*(T scalar)
  {
    mymatrix<T> result(NumRows, 1);
    int colSize;
    result.NumRows = this->NumRows;
    result.NumElements = this->NumElements;

    for (int row = 0; row < result.NumRows; row++){
      colSize = this->Rows[row].NumCols;
      result.Rows[row].Cols = new T[colSize]; //Array of type T with same number of elements as original
      result.Rows[row].NumCols = colSize;
      
      for (int col = 0; col < colSize; col++){
        result.Rows[row].Cols[col] = (this->Rows[row].Cols[col] * scalar); //Set value of each column element as original * scalar
      }
    }

    return result;
  }


  //
  // matrix multiplication
  //
  // Performs matrix multiplication M1 * M2, where M1 is "this" matrix and
  // M2 is the "other" matrix.  This produces a new matrix, which is returned.
  // "This" matrix is not changed, and neither is the "other" matrix.
  //
  // Example:  M3 = M1 * M2;
  //
  // NOTE: M1 and M2 must be rectangular, if not an exception is thrown.  In
  // addition, the sizes of M1 and M2 must be compatible in the following sense:
  // M1 must be of size RxN and M2 must be of size NxC.  In this case, matrix
  // multiplication can be performed, and the resulting matrix is of size RxC.
  //
  mymatrix<T> operator*(const mymatrix<T>& other)
  {
    if (!IsRectangular())
      throw runtime_error("mymatrix::*: this not rectangular");
    
    if (!other.IsRectangular())
      throw runtime_error("mymatrix::*: other not rectangular");

    if (this->numcols(0) != other.NumRows)
      throw runtime_error("mymatrix::*: size mismatch");

    mymatrix<T> result(this->NumRows, other.Rows[0].NumCols);
    for (int i = 0; i < result.NumRows; i++){
      for (int j = 0; j < result.numcols(0); j++){
        result.at(i,j) = CalcMatrixElement(this->Rows[i], other.Rows, j);
      }
    }

    return result;
  }


  //
  // _output
  //
  // Outputs the contents of the matrix; for debugging purposes.
  //
  void _output()
  {
    for (int r = 0; r < this->NumRows; ++r)
    {
      for (int c = 0; c < this->Rows[r].NumCols; ++c)
      {
        cout << this->Rows[r].Cols[c] << " ";
      }
      cout << endl;
    }
  }

};
