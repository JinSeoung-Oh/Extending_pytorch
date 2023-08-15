### Implementing and binding the class in C++

#include <torch/script.h>
#include <torch/custom_class.h>

#include <string>
#include <vector>

template <class T>
struct MyStackClass : torch::CustomClassHolder {
  std::vector<T> stack_;
  MyStackClass(std::vector<T> init) : stack_(init.begin(), init.end()) {}

  void push(T x) {
    stack_.push_back(x);
  }
  T pop() {
    auto val = stack_.back();
    stack_.pop_back();
    return val;
  }

  c10::intrusive_ptr<MyStackClass> clone() const {
    return c10::make_intrusive<MystackClass>(stack_);
  }

  void merge(const c10::intrusive_ptr<MyStackClass>& c) {
    for (auto& elem : c->stack_) {
      push(elem);
    }
  }
};
