.globl  _main
_add:
movq %r11, %rax
movq 0(%rax), %rax
movq 8(%rax), %rcx
pushq %rcx
movq %r11, %rax
movq 8(%rax), %rcx
popq %rax
addq %rax, %rcx
retq
_make_add:
movq _add@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
retq
_call_ec:
movq %r11, %rax
movq 8(%rax), %rcx
pushq %rcx
movq _make_ec@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
pushq %rcx
movq %rsp, %rcx
popq %rax
pushq %r11
movq 8(%rax), %r11
movq %r11, 0(%r10)
movq %rcx, 8(%r10)
movq %r10, %r11
addq $16, %r10
call *(%rax)
popq %r11
popq %rax
pushq %r11
movq 8(%rax), %r11
movq %r11, 0(%r10)
movq %rcx, 8(%r10)
movq %r10, %r11
addq $16, %r10
call *(%rax)
popq %r11
retq
_make_ec:
movq _ec@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
retq
_ec:
movq %r11, %rax
movq 0(%rax), %rax
movq 8(%rax), %rcx
movq %rcx, %rsp
movq %r11, %rax
movq 8(%rax), %rcx
retq
_lambda_0:
movq %r11, %rax
movq 8(%rax), %rcx
pushq %rcx
movq $10, %rcx
popq %rax
pushq %r11
movq 8(%rax), %r11
movq %r11, 0(%r10)
movq %rcx, 8(%r10)
movq %r10, %r11
addq $16, %r10
call *(%rax)
popq %r11
retq
_main:
movq _heap@GOTPCREL(%rip), %r10
movq _make_add@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
pushq %rcx
movq $1, %rcx
popq %rax
pushq %r11
movq 8(%rax), %r11
movq %r11, 0(%r10)
movq %rcx, 8(%r10)
movq %r10, %r11
addq $16, %r10
call *(%rax)
popq %r11
pushq %rcx
movq _call_ec@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
pushq %rcx
movq _lambda_0@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
popq %rax
pushq %r11
movq 8(%rax), %r11
movq %r11, 0(%r10)
movq %rcx, 8(%r10)
movq %r10, %r11
addq $16, %r10
call *(%rax)
popq %r11
popq %rax
pushq %r11
movq 8(%rax), %r11
movq %r11, 0(%r10)
movq %rcx, 8(%r10)
movq %r10, %r11
addq $16, %r10
call *(%rax)
popq %r11
movq %rcx, %rax
retq
.comm  _heap,536870912,4
