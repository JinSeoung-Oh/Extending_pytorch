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

### file structure
### cpp_inference_example/
###   infer.cpp
###   CMakeLists.txt
###   foo.pt
###   build/
###   custom_class_project/
###     class.cpp
###     CMakeLists.txt
###     build/

### infer.cpp

#include <torch/script.h>

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  torch::jit::Module module;
  try {
    module = torch::jit::load("foo.pt");
   }
  catch (const c10::Error& e) {
     std::cerr << "error loading the model\n";
     return -1;
   }

  std::vector<c10:IValue> inputs = {"foobarbaz"};
  auto output = module.forward(inputs).toString();
  std::cout << output->string() << std::endl;
 }

### CMakeLists.txt
cmake_minimum_requred(VERSION 3.1 FATAL_ERROR)
project(infer)

find_package(Torch REQUIRED)

add_subdirectory(custom_class_project)

add_executable(infer infer.cpp)
set(CMAKE_CXX_STANDARD 14)

target_link_libraries(infer "${TORCH_LIBRARIES}")
target_link_libraries(infer -Wl,--no-as-needed custom_class)


### Moving custorm classes TO/From IValues

see : https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html

### Defining Serialization/Deserialization Methods for Custom C++ Classes
### If try to save a ScriptModule with a custom-bound C++ class as an attribute, get the error
### This is because TorchScript cannot automatically figure out what information save from your C++ class
### The way to do that is to define __getstate__ and __setstate__ methods on the class using the special def_pickle method on class_.
### The semantics of __getstate__ and __setstate__ in TorchScript are equivalent to that of the Python pickle module. You can read more about how we use these methods.
### Add followe code in the MyStackClass

.def_pickle(
  [](const c10::intrusive_ptr<MyStackClass<std::string>>& self)
      -> std::vector<std::string> {
     return self->stack_;
  },
  [](std::vector<<std::string> state)
      -> c10::intrusive_ptr<MyStackClass<std::string>{
    return c10::make_intrusive<MyStackClass<std::string>>
(std::move(state));
});



### Defining Custom Operators that Take or Return Bound C++ Classes
### Once defined a custom C++ class, can also use that class as an argument or return from a custom operator (i.e. free functions)
c10::intrusive_ptr<MyStackClass<std::string>> manipulate_instance(const 
c10::intrusive_ptr<MyStackClass<std::string>>& instance) {
  instance->pop();
  return instance;
}

### then
m.def(
      "manipulate_instance(__torch__.torch.classes.my_classes.MyStackClass 
      x) -> __torch__.torch.classes.my_classes.MyStackClass Y",
      manipulate_instance
    );


### Once this is done, can use the op like the following example:

class TryCustomOp(torch.nn.Module):
    def __init__(self):
        super(TryCustomOp, self).__init__()
        self.f = torch.classes.my_classes.MyStackClass(["foo", "bar"])

    def forward(self):
        return torch.ops.my_classes.manipulate_instance(self.f)
