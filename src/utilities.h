#pragma once
#include <tuple>
#include <array>


// Utility to create a tuple of N repeated elements, equivalent to an std::array<T,N>.
// This tuple is useful for unpacking elements of an std::array<T,N> as function arguments.
namespace details
{
    template <typename T, typename... Ts>
    struct repeat_type;

    template <typename T, std::size_t... I>
    struct repeat_type<T, std::index_sequence<I...>>
    {
        // comma operator (I, T) discards the index_sequence in favour of T, but makes
        // the sequence a part of the expression so we can expand with ... thereby creating
        // N copies. std::decay_t is needed otherwise the expression can become an rvalue. 
        using type = std::tuple<std::decay_t<decltype(I, std::declval<T>())>...>;
    };
}

template <typename T, std::size_t N>
using repeat_type = typename details::repeat_type<T, std::make_index_sequence<N>>::type;


// A utility to count number of functor parameters.
template <typename F> struct Arity;
template <typename F, typename... Args>
struct Arity<F(*)(Args...)>
{
    static constexpr size_t value = sizeof...(Args);
};

/**
 * Utility to count the number of arguments returned by a function F, assuming it
 *   returns an std::array or an std::tuple.
 */
template <typename F, typename TupleType> struct Cardinality;
template <typename F, typename... Args>
struct Cardinality<F, std::tuple<Args...>>
{
    using ReturnType = decltype(std::apply(std::declval<F>(), std::declval<std::tuple<Args...>>()));
    static constexpr size_t value = std::tuple_size_v<std::decay_t<ReturnType>>;
};


namespace details
{
    template <typename T, std::size_t Start, std::size_t N, std::size_t... I, typename... Args>
    static std::array<T, N>
    unpack_as_array(std::index_sequence<I...>, Args&&... args)
    {
        return {{(std::get<Start + I>(std::forward_as_tuple(args...)))...}};
    }

    template <typename F, typename T, std::size_t N, std::size_t... I>
    static constexpr auto unpack_array(F&& func, const std::array<T, N>& arr,
                                       std::index_sequence<I...>)
    {
        return func(std::get<I>(arr)...);
    }
}

/**
 * Unpack argument lists as an std::array of type T.
 * The argument list must be the same type T (or convertable into T) for this to work.
 */
template <typename T, std::size_t Start, std::size_t N, typename... Args>
static auto unpack_as_array(Args&&... args)
{
    static_assert(sizeof...(Args) >= N + Start, "Not enough arguments provided.");
    return details::unpack_as_array<T, Start, N>(std::make_index_sequence<N>{},
                                                 std::forward<Args>(args)...);
}

/**
 * Unpack elements of an std::array as arguments to function F.
 * This allows for more flexible constructors that can take lists of Scalars rather
 * than having to cast them to an std::array.
 */
template <typename F, typename T, std::size_t N>
static constexpr auto unpack_array(F&& func, const std::array<T, N>& arr)
{
    return details::unpack_array(std::forward<F>(func), arr, std::make_index_sequence<N>{});
}

/// Helper to unpack arguments after an offset (Start) into a tuple
template <std::size_t Start, typename... Args>
static auto unpack_remaining(Args&&... args)
{
    static_assert(sizeof...(Args) >= Start, "Not enough arguments for unpacking remaining.");
    if constexpr (sizeof...(Args) > Start)
    {
        return std::get<Start>(std::forward_as_tuple(args...));
    }
    else return std::make_tuple();
}