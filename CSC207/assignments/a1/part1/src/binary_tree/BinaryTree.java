package binary_tree;

/**
 * A node in a binary tree with a value and left and right children.
 */
class BinaryNode {

    /** The value in this node. */
    Object value;

    /** The root of the left subtree. */
    BinaryNode left;

    /** The root of the right subtree. */
    BinaryNode right;

    /**
     * A new node with value v and no left or right children.
     *
     * @param v
     */
    BinaryNode(Object v) {
        this.value = v;
    }
}

/**
 * A binary tree of nodes.
 */
public class BinaryTree {

    /** The root of the tree. */
    private BinaryNode root;

    /**
     * Add a node for value under the node containing parentValue. Create a left
     * child if createLeftChild is true; otherwise create a right child. If a
     * node already exists at that location, replace it.
     *
     * If parentValue is null, then make the root of the tree a new node with no
     * children.
     *
     * Precondition: parentValue is null, or exists as a value somewhere in this
     * tree.
     *
     * @param value
     *            the value to add
     * @param parentValue
     *            where to add the new node
     * @param createLeftChild
     *            whether to create a left child or a right child
     */
    public void addValue(Object value, Object parentValue, boolean createLeftChild) {

        if (parentValue == null) {
            this.root = new BinaryNode(value);
        } else {

            // The current location in this tree.
            BinaryNode parent = findParent(root, parentValue);

            if (createLeftChild) {
                parent.left = new BinaryNode(value);
            } else {
                parent.right = new BinaryNode(value);
            }
        }
    }

    /**
     * Return the node containing parentValue, or null if the value is not in
     * the subtree rooted at curr. Note that curr may be null.
     *
     * @param curr
     *            the root of the subtree being searched.
     * @param parentValue
     *            the value being searched for.
     * @return the node containing parentValue, or null if the value is not in
     *         the subtree rooted at curr.
     */
    private BinaryNode findParent(BinaryNode curr, Object parentValue) {

        if (curr == null) {
            return null;
        } else {
    		if (curr.value == parentValue){
    			return curr;
    		} else if (curr.left != null){
        		return findParent(curr.left, parentValue);
            } else if (curr.right != null){
            	return findParent(curr.right, parentValue);
            } 
            return null;
        }

        // TODO: complete this method.
    }

    /**
     * Return a String representation of this tree of this form:
     *
     * "(rootvalue (left subtree) (right subtree))"
     *
     * or return "()" if the tree is empty. Empty subtrees should also be
     * represented as "()" — note that this means that all leaves will be
     * represented as "(value () ())".
     *
     * @return a String representation of this tree
     */
    public String toString() {
        if (this.root == null) {
            return "()";
        } else {
            return toString(this.root);
        }
    }

    /**
     * Return a String representation of the subtree rooted at curr of this
     * form:
     *
     * "(curr-value (left subtree) (right subtree))"
     *
     * or return "()" if the tree is empty. Empty subtrees should also be
     * represented as "()" — note that this means that all leaves will be
     * represented as "(curr-value () ())".
     *
     * @return a String representation of this tree
     */
    private String toString(BinaryNode curr) {
    	String s = new String();
    	
    	if (curr == null){
    		// current node is below a leaf 
    		return  "()";
    	} else {
    		// leaf or internal node 
    		s += "(" + curr.value + " " + toString(curr.left) + " " + toString(curr.right) + ")";
    	}
        return s;

        // TODO: complete this method.
    }

    public static void main(String[] args) {
        BinaryTree tree = new BinaryTree();
        System.out.println(tree);
        tree.addValue("A", null, false);
        System.out.println(tree);
        tree.addValue("B", "A", false);
        System.out.println(tree);
        tree.addValue("C", "B", true);
        System.out.println(tree);
        tree.addValue("D", "A", true);
        System.out.println(tree);
    }
}
