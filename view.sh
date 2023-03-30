#! /bin/bash

last_line=$(tail -1 $1)
printf "${last_line//$"  "/$"\n"}" 
