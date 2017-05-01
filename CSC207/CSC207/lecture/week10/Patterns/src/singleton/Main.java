package singleton;

public class Main {

    public static void main(String[] args) {
    
    PrintSpooler1 spooler1 = PrintSpooler1.getInstance();
    PrintSpooler1 spooler2 = PrintSpooler1.getInstance();
    System.out.println(spooler1 == spooler2);

    PrintSpooler2 spooler3 = PrintSpooler2.getInstance();
    PrintSpooler2 spooler4 = PrintSpooler2.getInstance();
    System.out.println(spooler3 == spooler4);

    }
}