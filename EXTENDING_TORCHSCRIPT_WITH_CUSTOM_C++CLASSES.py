### Implementing and binding the class in C++
### class.cpp

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


### 
TORCH_LIBRARY(my_classes, m){
  m.class_<MyStackClass<std::string>>("MyStackClass")
   .def(torch::init<std::vector<std::string>>())
   .def("top", [](const c10::intrusive_ptr<MyStackClass<std::string>>&
self){
      return self->stack_.back();
   })
   .def("push", &MyStackClass<std::string>::push)
   .def("pop", &MyStackClass<std::string>::pop)
   .def("clone", &MyStackClass<std::string>::clone)
   .def("merge", &MyStackClass<std::string>::merge)
;
}


### Building the Example as a C++ Project With CMake
### CMakeList.txt
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(custom_class)

find_package(Torch REQUIRED)

add_library(custom_class SHARED class.cpp)
set(CMAKE_CXX_STANDARD 14)

target_link_libraries(custom_class "&{TORCH_LIBRARIES}")

### folder structure
custom_class_project/
   class.cpp
   CMakeLists.txt
   build/

## After build, in the build direction, libcustom_class.so be generated

### Using the C++ Class from Python and TorchScript
import torch

torch.classes.load_library("build/libcustom_class.so")
s = torch.classes.my_classes.MyStackClass(["foo", "bar"])
s.push("pushed")
assert s.pop() == "pushed"

s.push("pushed")
torch.ops.my_classes.manipulate_instance(s)
assert s.top() == "bar"

s2 = s.clone()
s.merge(s2)
for expected in ["bar", "foo", "bar", "foo"]:
   assert s.pop() == expected

MyStackClass = torch.classes.my_classes.MyStackClass

@torch.jit.script
def do_stacks(s: MyStackClass):
    s2 = torch.classes.my_classes.MyStackClass(["hi", "mom"])
    s2.merge(s)
    return s2.clone(), s2.top()

stack, top = do_stacks(torch.classes.my_classes.MyStackClass(["wow"]))
assert top = "wow"
for expected in ["wow", "mom", "hi"]:
     assert stack.pop() == expected


### Saving, Loading, and Running TorchScript Code Using Custom Classes

import torch

torch.classes.load_library('build/libcustom_class.so')

class Foo(torch.nn.Module0:
     def __init__(self):
          super().__init__()
     def forward(self, s:str) -> str:
         stack = torch.classes.my_classes.MyStackClass(["hi", "mom"])
         return stack.pop() + s

scripted_foo = torch.jit.script(Foo())

scripted_foo.save('foo.pt')
