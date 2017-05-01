FIB:  addi $t3, $zero, 10        
      addi $t4, $zero, 1         
      addi $t5, $zero, -1        
LOOP: beq $t3, $zero, END        
      add $t4, $t4, $t5          
      sub $t5, $t4, $t5          
      addi $t3, $t3, -1
END: sb $t4, 0($sp)
      

