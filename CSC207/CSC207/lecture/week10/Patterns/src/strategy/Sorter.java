package strategy;

import java.util.List;

public interface Sorter<T> {
    
    /** Sorts the items in list in non-decreasing order. */
    public void sort(List<T> list);

}