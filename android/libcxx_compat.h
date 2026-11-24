// Force-included on Android: fills C++23 library gaps of the NDK r27 libc++ (LLVM 18).
#pragma once
#if defined(__cplusplus) && __cplusplus >= 202002L && defined(__ANDROID__)
#include <version>
#include <chrono>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#if !defined(__cpp_lib_move_only_function)
namespace std {
template <class>
class move_only_function;

template <class R, class... Args>
class move_only_function<R(Args...)> {
  struct Base {
    virtual ~Base() = default;
    virtual R Call(Args&&... args) = 0;
  };
  template <class F>
  struct Impl final : Base {
    F f;
    template <class G>
    explicit Impl(G&& g) : f(std::forward<G>(g)) {}
    R Call(Args&&... args) override { return std::invoke(f, std::forward<Args>(args)...); }
  };
  std::unique_ptr<Base> impl_;

 public:
  using result_type = R;
  move_only_function() noexcept = default;
  move_only_function(std::nullptr_t) noexcept {}
  move_only_function(move_only_function&&) noexcept = default;
  move_only_function& operator=(move_only_function&&) noexcept = default;
  move_only_function(const move_only_function&) = delete;
  move_only_function& operator=(const move_only_function&) = delete;
  move_only_function& operator=(std::nullptr_t) noexcept {
    impl_.reset();
    return *this;
  }
  template <class F, class D = std::decay_t<F>,
            class = std::enable_if_t<!std::is_same_v<D, move_only_function> &&
                                     !std::is_same_v<D, std::nullptr_t> &&
                                     std::is_invocable_r_v<R, D&, Args...>>>
  move_only_function(F&& f) {
    if constexpr (std::is_pointer_v<D> || std::is_member_pointer_v<D>) {
      if (!f) return;
    }
    impl_ = std::make_unique<Impl<D>>(std::forward<F>(f));
  }
  template <class F, class D = std::decay_t<F>,
            class = std::enable_if_t<!std::is_same_v<D, move_only_function> &&
                                     !std::is_same_v<D, std::nullptr_t> &&
                                     std::is_invocable_r_v<R, D&, Args...>>>
  move_only_function& operator=(F&& f) {
    move_only_function(std::forward<F>(f)).swap(*this);
    return *this;
  }
  void swap(move_only_function& o) noexcept { impl_.swap(o.impl_); }
  explicit operator bool() const noexcept { return impl_ != nullptr; }
  R operator()(Args... args) const { return impl_->Call(std::forward<Args>(args)...); }
  friend bool operator==(const move_only_function& f, std::nullptr_t) noexcept { return !f; }
};
}  // namespace std
#endif


#include <atomic>
#if !defined(__cpp_lib_atomic_ref)
namespace std {
// Minimal std::atomic_ref (libc++ 18 lacks it): what the SDK uses (load / store / exchange / fetch ops).
template <class T>
class atomic_ref {
 public:
  explicit atomic_ref(T& obj) noexcept : p_(&obj) {}
  T load(memory_order o = memory_order_seq_cst) const noexcept { return __atomic_load_n(p_, int(o)); }
  void store(T v, memory_order o = memory_order_seq_cst) const noexcept { __atomic_store_n(p_, v, int(o)); }
  T exchange(T v, memory_order o = memory_order_seq_cst) const noexcept { return __atomic_exchange_n(p_, v, int(o)); }
  T fetch_add(T v, memory_order o = memory_order_seq_cst) const noexcept { return __atomic_fetch_add(p_, v, int(o)); }
  T fetch_sub(T v, memory_order o = memory_order_seq_cst) const noexcept { return __atomic_fetch_sub(p_, v, int(o)); }
  T fetch_or(T v, memory_order o = memory_order_seq_cst) const noexcept { return __atomic_fetch_or(p_, v, int(o)); }
  T fetch_and(T v, memory_order o = memory_order_seq_cst) const noexcept { return __atomic_fetch_and(p_, v, int(o)); }
  bool compare_exchange_strong(T& e, T d, memory_order o = memory_order_seq_cst) const noexcept {
    return __atomic_compare_exchange_n(p_, &e, d, false, int(o), int(memory_order_relaxed));
  }
  operator T() const noexcept { return load(); }
 private:
  T* p_;
};
}  // namespace std
#endif
#endif
