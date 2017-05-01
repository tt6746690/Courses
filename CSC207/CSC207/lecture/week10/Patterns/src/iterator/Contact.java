package iterator;

/**
 * A contact in an address book, with last name, first name, email, phone, and
 * address.
 */
class Contact {

    /** This Contact's last name. */
    private String lastName;

    /** This Contact's first name. */
    private String firstName;

    /** This Contact's email address. */
    private String email;

    /** This Contact's phone number. */
    private String phone;

    /** This Contact's address. */
    private String address;

    /**
     * Constructs a new Contact with last name lastName, first name firstName,
     * email address email, phone number phone, and address address.
     * @param lastName the last name
     * @param firstName the first name
     * @param email the email address
     * @param phone the phone number
     * @param address the address
     */
    public Contact(String lastName, String firstName, String email,
            String phone, String address) {
        this.lastName = lastName;
        this.firstName = firstName;
        this.email = email;
        this.phone = phone;
        this.address = address;
    }

    /**
     * A comma-separated list of last name, first name, email address, phone
     * number, and address.
     * @return a comma-separated list of last name, first name, email address,
     * phone number, and address.
     */
    @Override
    public String toString() {
        return getLastName() + "," + getFirstName() + "," + getEmail() + ","
                + getPhone() + "," + getAddress();
    }

    /**
     * Returns this Contact's last name.
     * @return the last name
     */
    public String getLastName() {
        return lastName;
    }

    /**
     * Sets this Contact's last name to lastName.
     * @param lastName the lastName to set
     */
    public void setLastName(String lastName) {
        this.lastName = lastName;
    }

    /**
     * Returns this Contact's first name.
     * @return the first name
     */
    public String getFirstName() {
        return firstName;
    }

    /**
     * Sets this Contact's first name to firstName.
     * @param firstName the first name to set
     */
    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    /**
     * Returns this Contact's email address.
     * @return the email
     */
    public String getEmail() {
        return email;
    }

    /**
     * Sets this Contact's email address to email.
     * @param email the email to set
     */
    public void setEmail(String email) {
        this.email = email;
    }

    /**
     * Returns this Contact's phone number.
     * @return the phone
     */
    public String getPhone() {
        return phone;
    }

    /**
     * Sets this Contact's phone number to phone.
     * @param phone the phone to set
     */
    public void setPhone(String phone) {
        this.phone = phone;
    }

    /**
     * Returns this Contact's address.
     * @return the address
     */
    public String getAddress() {
        return address;
    }

    /**
     * Sets this Contact's address to address.
     * @param address the address to set
     */
    public void setAddress(String address) {
        this.address = address;
    }
}