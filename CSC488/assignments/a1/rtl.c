#include <stdio.h>

/*
#;(L2: closure <name>) ; closure(<name>);
#;(L2: variable <n>) ; variable(<n>);
#;(L2: push_result) ; push_result();
#;(L2: call) ; call();
#;(L2: set <n>) ; set(<n>);
#;(L2: set_result <i>) ; set_result(<i>);
*/

void* heap[1073741824];  // array of pointers 
void** next = heap;      // pointer to elements in the array

void* stack[65536];      // array of pointers 
void** top = stack;      // pointer to elements in the array, top means first 

void** env = heap;       // pointer to current env
void* result;            // a global variable 

// C call stack usage is shallow, these are easily inlined.
void push(void* v) { 
    printf("push(%lld) stack[%ld]\n", (long long)v, top-stack);
    top[0] = v; top += 1; 
}
void** pop() { 
    top -= 1; 
    printf("pop(%lld)  stack[%ld]\n", (long long)(top[0]), top-stack);
    return (void**)(top[0]); 
}

// store [first, second] at location of next on heap, 
// return pointer to [first, second]
void** pair(void* first, void* second) {
    next[0] = first; next[1] = second;
    printf("pair(%lld, %lld) at heap[%ld, %ld]\n", (long long)first, (long long)second, next-heap, next+1-heap);
    next += 2; 
    return next - 2; 
}
void** environment(void* e, void* v) { 
    printf("environment(%lld, %lld)\n", (long long)e, (long long)v);
    return pair(e, v); 
}

// Results via global result value.
// Tail wrt any local variables, including in client code.
// Thus easily inlined.

/*
(λ (id) e)
     Add a closure λ<n> pairing this λ expression and current environment, to the set of closures.
     Set the current result to be λ<n>.
*/

// Closure takes in a pointer to a lambda, 
// and store the lambda pointer and current env as its parent on heap

// Create a unary closure from the compiled body
// Leave a reference to it in result.
void closure(void body()) {
    printf("closure(%lld)\n", (void*)body);
    result = pair((void*)body, env); 
}

// push result to stack
void push_result() { 
    printf("push_result(%lld)\n", (long long)result);
    push(result); 
}

/* 
   id
     Valid only during the evaluation of a λ body, during a call to the λ, where id is one of
      the parameters in the chain of environments from the closure's environment upwards.
     Set the current result to be the value of id in that environment.
*/
// Look up the variable <n> environments up from the current environment, put its value in result.
void variable(int n) {
    void** env_temp = env;
    while (n != 0) { 
        env_temp = (void**)(env_temp[0]); --n; 
    }
    result = env_temp[1]; 
}

/* 
(e1 e2)
     Evaluate e1.
     Push the current result [the value of e1] onto the stack of results.
     Evaluate e2. (note current result now is value of e2)

     Pop to get the closure to call, let's refer to it as λf.
     Add a new environment E<n> to the tree of environments, under λf's environment, with the id
      from λf's λ expression and the current result [which is the value of e2].
     Push the current environment onto the call stack.
     Set the current environment to E<n>.
     Evaluate the body of λf's λ expression.
     Pop the call stack into the current environment. 
*/
void call() {
    void** f = pop();
    push(env);
    env = environment(f[1], result);  // f[1] is a closure (pointer to f, env of f)
    ((void(*)())(f[0]))(); // call f
    env = pop(); 
}

// set_result to integer i
void set_result(long long i) {
    result = (void*)i;
}

// Set the variable n environments up from the current environment, to the value of result.
void set(int n) {
    void** env_temp = env;
    while (n != 0) { 
        env_temp = (void**)(env_temp[0]); --n; 
    }
    env_temp[1] = result;
}

void add() {
  variable(1);
  long long temp = (long long)result;
  variable(0);
  result = (void*)(temp + (long long)result); 
}

void make_add() { 
    closure(add); 
}

void multiply() {
    variable(1);
    long long temp = (long long)result;
    variable(0);
    printf("multiply %lld * %lld \n", temp, (long long)result);
    result = (void*)(temp * (long long) result);
}

void make_multiply() {
    closure(multiply);
}



// Similarly, add make_less_than to the runtime library. Based on the comparison of the two
// [implicit] arguments, the result should be a closure representing the Church encoding of
// true or false: a curried binary functions that calls its first or second argument with a
// simple dummy value.

void bool_true_inner() {
  variable(1);
}
void bool_true() {
  closure(bool_true_inner);
}
void bool_false_inner() {
  variable(0);
}
void bool_false() {
  closure(bool_false_inner);
}


void less_than() {
  variable(1);
  long long temp = (long long)result;
  variable(0);
  printf("less_than %lld < %lld ", temp, (long long)result);
  if(temp < (long long)result) { 
    printf("true\n"); 
    closure(bool_true);
  } else {
    printf("false\n");
    closure(bool_false);
  }
}


void make_less_than() {
    closure(less_than);
}



// ((λ (x) (set! x 12)) 2 )
// void lambda_0() {  
//     set_result(12);
//     set(0);
// }
// int main() {
//     closure(lambda_0);  
//     push_result();  
//     set_result(2);  
//     call();  
//     printf("%lld", result);  
//     return 0;
// }


// (+ 1 2)
// int main() {
//   closure(make_add);
//   push_result();
//   set_result(1);
//   call();
//   push_result();
//   set_result(2);
//   call();
//   printf("%lld\n", result);
//   return 0;
// }

// (* 4 20)
// int main() {
//   closure(make_multiply);
//   push_result();
//   set_result(4);
//   call();
//   push_result();
//   set_result(20);
//   call();
//   printf("%lld\n", result);
//   return 0;
// }



// true encoding 

// void lambda_0() {
//     variable(1);
// }
// void lambda_1() {
//     closure(lambda_0);
// }
// int main() {
//     closure(lambda_1);
//     printf("%lld\n", result);
//     return 0;
// }

// (< 1 0)
// int main() {
//     closure(make_less_than);
//     push_result();
//     set_result(1);
//     call();
//     push_result();
//     set_result(0);
//     call();
//    return 0;
// }