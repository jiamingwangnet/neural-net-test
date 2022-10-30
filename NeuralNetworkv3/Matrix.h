#pragma once

#include <vector>
#include <iterator>
#include <cassert>

namespace math
{
	template<typename T>
	class Matrix
	{
	public:
		Matrix(std::vector<T> values, size_t rows, size_t columns, T init = 0);
		Matrix(size_t rows, size_t columns, T init = 0);
		Matrix();
	public:
		typename std::vector<T>::iterator begin(); // typename is there because the type of the iterator is unknown.
		typename std::vector<T>::iterator end();
		const typename std::vector<T>::const_iterator begin() const;
		const typename std::vector<T>::const_iterator end() const;
	public:
		T& operator()(size_t row, size_t column);
		const T& operator()(size_t row, size_t column) const;
		T& operator[](size_t index);
		const T& operator[](size_t index) const;

		Matrix operator*(const Matrix& rhs) const;
		Matrix operator*=(const Matrix& rhs);
		Matrix operator*(const T& rhs) const;
		Matrix operator*=(const T& rhs);

		Matrix operator+(const Matrix& rhs) const;
		Matrix operator+=(const Matrix& rhs);
		Matrix operator-(const Matrix& rhs) const;
		Matrix operator-=(const Matrix& rhs);

		bool operator==(const Matrix& rhs) const;
		bool operator!=(const Matrix& rhs) const;

		Matrix Hadamard(const Matrix& rhs) const;
		Matrix GetTransposed() const;
		
		bool SizeEqu(const Matrix& other) const;
	public:
		 size_t GetRows() const;
		 size_t GetColumns() const;
		 size_t GetSize() const;
	private:
		std::vector<T> values;
		size_t rows;
		size_t columns;
	};
	
	template<typename T>
	inline math::Matrix<T>::Matrix(std::vector<T> values, size_t rows, size_t columns, T init)
		: values(values), rows(rows), columns(columns)
	{
		values.resize(rows * columns, init);
	}

	template<typename T>
	inline math::Matrix<T>::Matrix(size_t rows, size_t columns, T init)
		: rows(rows), columns(columns)
	{
		values.resize(rows * columns, init);
	}

	template<typename T>
	inline math::Matrix<T>::Matrix()
		: rows(0), columns(0)
	{}

	template<typename T>
	inline typename std::vector<T>::iterator math::Matrix<T>::begin()
	{
		return values.begin();
	}

	template<typename T>
	inline typename std::vector<T>::iterator math::Matrix<T>::end()
	{
		return values.end();
	}

	template<typename T>
	inline const typename std::vector<T>::const_iterator math::Matrix<T>::begin() const
	{
		return values.begin();
	}

	template<typename T>
	inline const typename std::vector<T>::const_iterator math::Matrix<T>::end() const
	{
		return values.end();
	}

	template<typename T>
	inline T& math::Matrix<T>::operator()(size_t row, size_t column)
	{
		return values[row * columns + column];
	}

	template<typename T>
	inline const T& math::Matrix<T>::operator()(size_t row, size_t column) const
	{
		return values[row * columns + column];
	}

	template<typename T>
	inline T& math::Matrix<T>::operator[](size_t index)
	{
		return values[index];
	}

	template<typename T>
	inline const T& math::Matrix<T>::operator[](size_t index) const
	{
		return values[index];
	}

	template<typename T>
	inline math::Matrix<T> math::Matrix<T>::operator*(const Matrix& rhs) const
	{
		assert(columns == rhs.rows);
		Matrix res{ rows, rhs.columns };
		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < rhs.columns; j++)
			{
				for (size_t k = 0; k < rhs.rows; k++)
				{
					res(i, j) += (*this)(i, k) * rhs(k, j);
				}
			}
		}
		return res;
	}

	template<typename T>
	inline math::Matrix<T> math::Matrix<T>::operator*=(const Matrix& rhs)
	{
		return (*this) = (*this) * rhs;
	}

	template<typename T>
	inline math::Matrix<T> math::Matrix<T>::operator*(const T& rhs) const
	{
		Matrix res{ rows, columns };
		for (size_t i = 0; i < rows * columns; i++)
		{
			res[i] = (*this)[i] * rhs;
		}
		return res;
	}

	template<typename T>
	inline math::Matrix<T> math::Matrix<T>::operator*=(const T& rhs)
	{
		return (*this) = (*this) * rhs;
	}

	template<typename T>
	inline math::Matrix<T> math::Matrix<T>::operator+(const Matrix& rhs) const
	{
		assert(SizeEqu(rhs));
		Matrix res{ rows, columns };
		for (size_t i = 0; i < rows * columns; i++)
		{
			res[i] = (*this)[i] + rhs[i];
		}
		return res;
	}

	template<typename T>
	inline math::Matrix<T> math::Matrix<T>::operator+=(const Matrix& rhs)
	{
		return (*this) = *(this) + rhs;
	}

	template<typename T>
	inline math::Matrix<T> math::Matrix<T>::operator-(const Matrix& rhs) const
	{
		assert(SizeEqu(rhs));
		Matrix res{ rows, columns };
		for (size_t i = 0; i < rows * columns; i++)
		{
			res[i] = (*this)[i] - rhs[i];
		}
		return res;
	}

	template<typename T>
	inline math::Matrix<T> math::Matrix<T>::operator-=(const Matrix& rhs)
	{
		return(*this) = (*this) - rhs;
	}

	template<typename T>
	inline bool math::Matrix<T>::operator==(const Matrix& rhs) const
	{
		return values == rhs.values && SizeEqu(rhs);
	}

	template<typename T>
	inline bool math::Matrix<T>::operator!=(const Matrix& rhs) const
	{
		return !(*this == rhs);
	}

	template<typename T>
	inline math::Matrix<T> math::Matrix<T>::Hadamard(const Matrix& rhs) const
	{
		assert(SizeEqu(rhs));
		Matrix res{ rows, columns };
		for (size_t i = 0; i < rows * columns; i++)
		{
			res[i] = (*this)[i] * rhs[i];
		}
		return res;
	}

	template<typename T>
	inline math::Matrix<T> math::Matrix<T>::GetTransposed() const
	{
		Matrix res{ columns, rows };
		for (size_t r = 0; r < rows; r++)
		{
			for (size_t c = 0; c < columns; c++)
			{
				res(c, r) = (*this)(r, c);
			}
		}
		return res;
	}

	template<typename T>
	inline bool math::Matrix<T>::SizeEqu(const Matrix& other) const
	{
		return rows == other.rows && columns == other.columns;
	}

	template<typename T>
	inline size_t math::Matrix<T>::GetRows() const
	{
		return rows;
	}

	template<typename T>
	inline size_t math::Matrix<T>::GetColumns() const
	{
		return columns;
	}

	template<typename T>
	inline size_t math::Matrix<T>::GetSize() const
	{
		return rows * columns;
	}

	typedef Matrix<double> DMatrix;
}