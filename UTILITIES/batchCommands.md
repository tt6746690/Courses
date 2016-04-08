[DOS COMMANDS] (https://technet.microsoft.com/en-us/library/bb490882.aspx)
===========

#### Tricks


`ECHO %CD%` - return current working directory


- **

#### Commands

__Info about commands__   
`HELP [command]`  

__Rename files__     
`REN [drive:][path]filename1 filename2`  

__Moving files__    
`MOVE [/Y | /-Y] [drive:][path]filename1[,...] destination`

__Delete files__  
`DEL [/P] [/F] [/S] [/Q] [/A[[:]attributes]] names`

__File strings__  
`FINDSTR [/B] [/E] [/L] [/R] [/S] [/I] [/X] [/V] [/N] [/M] [/O] [/P] [/F:file]
        [/C:string] [/G:file] [/D:dir list] [/A:color attributes] [/OFF[LINE]]
        strings [[drive:][path]filename[ ...]]`     

`/R` search string using regular expression     
`/S` search recursively  
`/I` search case insensitive  
`/V` invert search result to output that do not match  
`/M` prints only the filename  

__IPCONFIG__  

__NETSTAT__

__PING__

__SHUTDOWN__  
`shutdown [/i | /l | /s | /r | /g | /a | /p | /h | /e | /o] [/hybrid] [/f] [/m \\computer][/t xxx][/d [p|u:]xx:yy [/c "comment"]]`  

`/S` shut down computer  
`/R` full shutdown and restart  
`/O` shutdown and restart to advanced menu    

[__TASKLIST__] (https://technet.microsoft.com/en-ca/library/bb491010.aspx)  
`TASKLIST [/S system [/U username [/P [password]]]] [/M [module] | /SVC | /V] [/FI filter] [/FO format] [/NH]`   

<tbody><tr>
    <th>Filter Name</th>
    <th>Valid Operators</th>
    <th>Valid values</th>
  </tr>
  <tr class="tcw">
    <td>STATUS</td>
    <td>eq, ne</td>
    <td>RUNNING | NOT RESPONDING | UNKNOWN</td>
  </tr>
  <tr class="tcw">
    <td>IMAGENAME</td>
    <td>eq, ne</td>
    <td>Image name</td>
  </tr>
  <tr class="tcw">
    <td><a href="jargon/p/pid.htm">PID</a></td>
    <td>eq, ne, gt, lt, ge, le</td>
    <td>PID value</td>
  </tr>
  <tr class="tcw">
    <td>SESSION</td>
    <td>eq, ne, gt, lt, ge, le</td>
    <td>Session number</td>
  </tr>
  <tr class="tcw">
    <td>SESSIONNAME</td>
    <td>eq, ne</td>
    <td>Session name</td>
  </tr>
  <tr class="tcw">
    <td>CPUTIME</td>
    <td>eq, ne, gt, lt, ge, le</td>
    <td>CPU time in the format of hh:mm:ss.<br>
      hh - hours,<br>
      mm - minutes<br>
      ss - seconds</td>
  </tr>
  <tr class="tcw">
    <td>MEMUSAGE</td>
    <td>eq, ne, gt, lt, ge, le</td>
    <td>Memory usage in KB</td>
  </tr>
  <tr class="tcw">
    <td>USERNAME</td>
    <td>eq, ne</td>
    <td>Username in [domain\]user format</td>
  </tr>
  <tr class="tcw">
    <td>SERVICES</td>
    <td>eq, ne</td>
    <td>Service name</td>
  </tr>
  <tr class="tcw">
    <td>WINDOWTITLE</td>
    <td>eq, ne</td>
    <td>Window title</td>
  </tr>
  <tr class="tcw">
    <td>MODULES</td>
    <td>eq, ne</td>
    <td>DLL name</td>
  </tr>
</tbody>


_Example_:  
`tasklist /fi "memusage gt 50000"`

TASKKILL


- **

[__Useful CMD shortcuts__] (http://ss64.com/nt/syntax-keyboard.html)
ALT+ENTER  _Switch to/from full screen mode._  
  [Tab]    Autocomplete folder/file name.  
  ↓ / ↑    Scroll through history of typed commands.  
 F1 F1 F1  Print characters of the previous command one by one.  
  F2 Z     Repeat part of the previous command; up to character Z  
  F3       Repeat the previous command.   
  F4 Z     Beginning from the current cursor position, delete up to character Z.  
  F5       Scroll through history of typed commands (↑).  
  F7       Show history of previous commands.  
 ALT+F7    Clear command history.  
  F8       _Move backwards through the command history, but only display  
           commands matching the current text at the command prompt._
  F9       Run a specific command from the command history.  
  ESC      Clear command line.  
 INSERT    Toggle Insert/Overwrite.  
 Ctrl Home Erase line to the left.  
 Ctrl End  Erase line to the right.  
 Ctrl ←    Move one word to the left (backward).  
 Ctrl →    Move one word to the right (forward).  
  ⌫       Erase character to the left.   
 [Home]    Move to beginning of line.  
 [End]     Move to end of line.  
  ⇧ PgUp   Scroll window up.  
  ⇧ PgDn   Scroll window Down.  
 Ctrl-C    Abort current command/typing.  
Left Alt + Left ⇧ + PrtScn  
           Toggle High Visibility screen mode.  
 Ctrl-Z    Signal end-of-file.  
