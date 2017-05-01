int main() {

    int i = 81;
    int *pt = &i;
    int **pt_ptr = &pt;		// pt_ptr is a pointer to a pointer 

    int *r= *pt_ptr;		// dereferencing pt_ptr gives address stored in pt, i.e. pt
    int k = *r;			// k is 81
   
    // We don't actually need the intermediate value r. 
    // We can dereference pt_ptr twice like this.
    int k1 = **pt_ptr;
  
    // We can even have triple pointers.
    int ***pt_ptr_ptr = &pt_ptr;
    int k2 = ***pt_ptr_ptr;
    return 0;
}
