#ifndef INDEXED_VECTOR_HPP
#define	INDEXED_VECTOR_HPP

#include	<allocators>
#include	<vector>

template<unsigned int Idx, typename T, typename Allocator = std::allocator<T> >
class indexed_vector : public std::vector<T>
{
private:
	typedef std::vector<T> base;

public:
	explicit indexed_vector(const Allocator& alloc = Allocator())
		: base(alloc)
	{ }

	indexed_vector(size_type count, const T& value, const Allocator& alloc = Allocator())
		: base(count, value, alloc)
	{ }

	explicit indexed_vector(size_type count)
		: base(count)
	{ }

	template<class InputIt>
	indexed_vector(InputIt first, InputIt last, const Allocator& alloc = Allocator())
		: base(first, last, alloc)
	{ }

	indexed_vector(const indexed_vector& other)
		: base(other)
	{ }

	indexed_vector(const base& other)
		: base(other)
	{ }

	indexed_vector(const indexed_vector& other, const Allocator& alloc)
		: base(other, alloc)
	{ }

	indexed_vector(const base& other, const Allocator& alloc)
		: base(other, alloc)
	{ }

	indexed_vector(indexed_vector&& other)
		: base(other)
	{ }

	indexed_vector(base&& other)
		: base(other)
	{ }

	indexed_vector(indexed_vector&& other, const Allocator& alloc)
		: base(other, alloc)
	{ }

	indexed_vector(base&& other, const Allocator& alloc)
		: base(other, alloc)
	{ }


	indexed_vector(std::initializer_list<T> init, const Allocator& alloc = Allocator())
		: base(init, alloc)
	{ }

	indexed_vector& operator=(const base& other)
	{
		base::operator=(other);
		return *this;
	}

	indexed_vector& operator=(const indexed_vector& other)
	{
		base::operator=(other);
		return *this;
	}

	~indexed_vector()
	{ }

	reference at(size_type pos)
	{
		return this->base::at(pos - Idx);
	}

	const_reference at(size_type pos) const
	{
		return this->base::at(pos - Idx);
	}

	reference operator[](size_type pos)
	{
		return this->base::operator[](pos - Idx);
	}

	const_reference operator[](size_type pos) const
	{
		return this->base::operator[](pos - Idx);
	}
};

#endif // !INDEXED_VECTOR_HPP
