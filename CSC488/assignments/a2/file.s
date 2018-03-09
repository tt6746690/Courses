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
_multiply:
movq %r11, %rax
movq 0(%rax), %rax
movq 8(%rax), %rcx
pushq %rcx
movq %r11, %rax
movq 8(%rax), %rcx
popq %rax
imulq %rax, %rcx
retq
_make_multiply:
movq _multiply@GOTPCREL(%rip), %rax
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
_less_than:
movq %r11, %rax
movq 0(%rax), %rax
movq 8(%rax), %rcx
pushq %rcx
movq %r11, %rax
movq 8(%rax), %rcx
popq %rax
cmpq %rcx, %rax
setb %cl
movzbq %cl, %rcx
retq
_make_less_than:
movq _less_than@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
retq
_lambda_24:
movq _lambda_23@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
retq
_lambda_23:
movq _lambda_22@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
retq
_lambda_22:
movq _lambda_21@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
retq
_lambda_21:
movq _lambda_20@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
retq
_lambda_20:
movq _lambda_19@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
retq
_lambda_19:
movq _lambda_17@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
pushq %rcx
movq _lambda_18@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
movq %r11, %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq %rcx, 8(%rax)
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
_lambda_17:
movq _lambda_14@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
pushq %rcx
movq _lambda_16@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
movq %r11, %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq %rcx, 8(%rax)
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
_lambda_14:
movq _lambda_12@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
pushq %rcx
movq _lambda_13@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
movq %r11, %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq %rcx, 8(%rax)
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
_lambda_12:
movq _lambda_9@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
pushq %rcx
movq _lambda_11@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
movq %r11, %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq %rcx, 8(%rax)
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
_lambda_9:
movq _lambda_6@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
pushq %rcx
movq _lambda_8@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
movq %r11, %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq %rcx, 8(%rax)
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
_lambda_6:
movq _lambda_3@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
pushq %rcx
movq _lambda_5@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
movq %r11, %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq %rcx, 8(%rax)
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
_lambda_3:
movq _lambda_2@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
pushq %rcx
movq $0, %rcx
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
_lambda_2:
movq _lambda_0@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
pushq %rcx
movq _lambda_1@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
movq %r11, %rax
movq %rcx, 8(%rax)
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
_lambda_0:
movq %r11, %rax
movq 0(%rax), %rax
movq 8(%rax), %rcx
pushq %rcx
movq $13, %rcx
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
_lambda_1:
movq %r11, %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 8(%rax), %rcx
pushq %rcx
movq %r11, %rax
movq 8(%rax), %rcx
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
cmpq $0, %rcx
je _else_0
movq $1, %rcx
jmp _end_0
_else_0:
movq %r11, %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 8(%rax), %rcx
pushq %rcx
movq %r11, %rax
movq 8(%rax), %rcx
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
movq $2, %rcx
popq %rax
pushq %r11
movq 8(%rax), %r11
movq %r11, 0(%r10)
movq %rcx, 8(%r10)
movq %r10, %r11
addq $16, %r10
call *(%rax)
popq %r11
cmpq $0, %rcx
je _else_1
movq $1, %rcx
jmp _end_1
_else_1:
movq _make_add@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
pushq %rcx
movq %r11, %rax
movq 0(%rax), %rax
movq 8(%rax), %rcx
pushq %rcx
movq %r11, %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 8(%rax), %rcx
pushq %rcx
movq %r11, %rax
movq 8(%rax), %rcx
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
pushq %rcx
movq %r11, %rax
movq 0(%rax), %rax
movq 8(%rax), %rcx
pushq %rcx
movq %r11, %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 8(%rax), %rcx
pushq %rcx
movq %r11, %rax
movq 8(%rax), %rcx
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
movq $2, %rcx
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
popq %rax
pushq %r11
movq 8(%rax), %r11
movq %r11, 0(%r10)
movq %rcx, 8(%r10)
movq %r10, %r11
addq $16, %r10
call *(%rax)
popq %r11
_end_1:
_end_0:
retq
_lambda_5:
movq _lambda_4@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
retq
_lambda_4:
movq %r11, %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 8(%rax), %rcx
pushq %rcx
movq %r11, %rax
movq 0(%rax), %rax
movq 8(%rax), %rcx
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
movq %r11, %rax
movq 8(%rax), %rcx
popq %rax
pushq %r11
movq 8(%rax), %r11
movq %r11, 0(%r10)
movq %rcx, 8(%r10)
movq %r10, %r11
addq $16, %r10
call *(%rax)
popq %r11
cmpq $0, %rcx
je _else_2
movq %r11, %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 8(%rax), %rcx
pushq %rcx
movq %r11, %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 8(%rax), %rcx
pushq %rcx
movq %r11, %rax
movq 0(%rax), %rax
movq 8(%rax), %rcx
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
movq %r11, %rax
movq 8(%rax), %rcx
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
cmpq $0, %rcx
je _else_3
movq $1, %rcx
jmp _end_3
_else_3:
movq $0, %rcx
_end_3:
jmp _end_2
_else_2:
movq $0, %rcx
_end_2:
retq
_lambda_8:
movq _lambda_7@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
retq
_lambda_7:
movq %r11, %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 8(%rax), %rcx
pushq %rcx
movq _make_less_than@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
pushq %rcx
movq %r11, %rax
movq 0(%rax), %rax
movq 8(%rax), %rcx
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
movq %r11, %rax
movq 8(%rax), %rcx
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
_lambda_11:
movq _lambda_10@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
retq
_lambda_10:
movq _make_less_than@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
pushq %rcx
movq %r11, %rax
movq 8(%rax), %rcx
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
movq %r11, %rax
movq 0(%rax), %rax
movq 8(%rax), %rcx
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
_lambda_13:
movq _make_multiply@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
pushq %rcx
movq $-1, %rcx
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
movq %r11, %rax
movq 8(%rax), %rcx
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
_lambda_16:
movq _lambda_15@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
retq
_lambda_15:
movq _make_add@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
pushq %rcx
movq %r11, %rax
movq 0(%rax), %rax
movq 8(%rax), %rcx
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
movq %r11, %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 0(%rax), %rax
movq 8(%rax), %rcx
pushq %rcx
movq %r11, %rax
movq 8(%rax), %rcx
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
_lambda_18:
movq %r11, %rax
movq 8(%rax), %rcx
cmpq $0, %rcx
je _else_4
movq $0, %rcx
jmp _end_4
_else_4:
movq $1, %rcx
_end_4:
retq
_main:
movq _heap@GOTPCREL(%rip), %r10
movq _lambda_24@GOTPCREL(%rip), %rax
movq %rax, 0(%r10)
movq %r11, 8(%r10)
movq %r10, %rcx
addq $16, %r10
pushq %rcx
movq $0, %rcx
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
movq $0, %rcx
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
movq $0, %rcx
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
movq $0, %rcx
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
movq $0, %rcx
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
movq $0, %rcx
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
