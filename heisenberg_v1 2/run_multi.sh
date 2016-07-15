#!/bin/sh

INPUT="input"
OUTFILE="out"
DATA="tmp_jrs.dat jrs.omni mfile.dat file.dat"


ID=$SLURM_PROCID
NP=$SLURM_STEP_NUM_TASKS
if [ -z "$ID" ] ; then ID=0; fi
if [ -z "$NP" ] ; then NP=1; fi

JOBDIR=`echo $ID | awk '{printf("job-%04d",$1+1)}'`

# creating a uniq directory for each process
mkdir -p $JOBDIR
cd $JOBDIR

# creating link to data file with J's etc
for i in $DATA ; do
   if [ ! -e $i ] ; then ln -s ../$i . ; fi
done




###########################
# create input files, all answers the program will ask, ID starts with 0!
###########################
case "$ID" in                                       
 0)
      echo "2"      > $INPUT    # repetition in x,y,z                                                           
      echo "1000"  >> $INPUT    # MC steps                                                                      
      echo "0.0"   >> $INPUT    # H field                                                                       
      echo "60"    >> $INPUT    # temperature                                                                   
      echo "1.9785"  >> $INPUT # local effective magnetic moment of Fe                                          
      echo "-0.1572"  >> $INPUT # local magnetic moment of B                                                    
      ;;
 1)
      echo "2"      > $INPUT    # repetition in x,y,z                                                           
      echo "1000"  >> $INPUT    # MC steps                                                                      
      echo "0.0"   >> $INPUT    # H field                                                                       
      echo "55"    >> $INPUT    # temperature                                                                   
      echo "1.9785"  >> $INPUT # local effective magnetic moment of Fe                                          
      echo "-0.1572"  >> $INPUT # local magnetic moment of B                                                    
      ;;
 2)
      echo "2"      > $INPUT    # repetition in x,y,z                                                           
      echo "1000"  >> $INPUT    # MC steps                                                                      
      echo "0.0"   >> $INPUT    # H field                                                                       
      echo "50"    >> $INPUT    # temperature                                                                   
      echo "1.9785"  >> $INPUT # local effective magnetic moment of Fe                                          
      echo "-0.1572"  >> $INPUT # local magnetic moment of B                                                    
      ;;
 3)
      echo "2"      > $INPUT    # repetition in x,y,z                                                           
      echo "1000"  >> $INPUT    # MC steps                                                                      
      echo "0.0"   >> $INPUT    # H field                                                                       
      echo "45"    >> $INPUT    # temperature                                                                   
      echo "1.9785"  >> $INPUT # local effective magnetic moment of Fe                                          
      echo "-0.1572"  >> $INPUT # local magnetic moment of B                                                    
      ;;
 4)
      echo "2"      > $INPUT    # repetition in x,y,z                                                           
      echo "1000"  >> $INPUT    # MC steps                                                                      
      echo "0.0"   >> $INPUT    # H field                                                                       
      echo "40"    >> $INPUT    # temperature                                                                   
      echo "1.9785"  >> $INPUT # local effective magnetic moment of Fe                                          
      echo "-0.1572"  >> $INPUT # local magnetic moment of B                                                    
      ;;
 5)
      echo "2"      > $INPUT    # repetition in x,y,z                                                           
      echo "1000"  >> $INPUT    # MC steps                                                                      
      echo "0.0"   >> $INPUT    # H field                                                                       
      echo "35"    >> $INPUT    # temperature                                                                   
      echo "1.9785"  >> $INPUT # local effective magnetic moment of Fe                                          
      echo "-0.1572"  >> $INPUT # local magnetic moment of B                                                    
      ;;
 6)
      echo "2"      > $INPUT    # repetition in x,y,z                                                           
      echo "1000"  >> $INPUT    # MC steps                                                                      
      echo "0.0"   >> $INPUT    # H field                                                                       
      echo "30"    >> $INPUT    # temperature                                                                   
      echo "1.9785"  >> $INPUT # local effective magnetic moment of Fe                                          
      echo "-0.1572"  >> $INPUT # local magnetic moment of B                                                    
      ;;
 7)
      echo "2"      > $INPUT    # repetition in x,y,z                                                           
      echo "1000"  >> $INPUT    # MC steps                                                                      
      echo "0.0"   >> $INPUT    # H field                                                                       
      echo "25"    >> $INPUT    # temperature                                                                   
      echo "1.9785"  >> $INPUT # local effective magnetic moment of Fe                                          
      echo "-0.1572"  >> $INPUT # local magnetic moment of B                                                    
      ;;
 8)
      echo "2"      > $INPUT    # repetition in x,y,z                                                           
      echo "1000"  >> $INPUT    # MC steps                                                                      
      echo "0.0"   >> $INPUT    # H field                                                                       
      echo "20"    >> $INPUT    # temperature                                                                   
      echo "1.9785"  >> $INPUT # local effective magnetic moment of Fe                                          
      echo "-0.1572"  >> $INPUT # local magnetic moment of B                                                    
      ;;
 9)
      echo "2"      > $INPUT    # repetition in x,y,z                                                           
      echo "1000"  >> $INPUT    # MC steps                                                                      
      echo "0.0"   >> $INPUT    # H field                                                                       
      echo "15"    >> $INPUT    # temperature                                                                   
      echo "1.9785"  >> $INPUT # local effective magnetic moment of Fe                                          
      echo "-0.1572"  >> $INPUT # local magnetic moment of B                                                    
      ;;
 10)
      echo "2"      > $INPUT    # repetition in x,y,z                                                           
      echo "1000"  >> $INPUT    # MC steps                                                                      
      echo "0.0"   >> $INPUT    # H field                                                                       
      echo "10"    >> $INPUT    # temperature                                                                   
      echo "1.9785"  >> $INPUT # local effective magnetic moment of Fe                                          
      echo "-0.1572"  >> $INPUT # local magnetic moment of B                                                    
      ;;
 11)
      echo "2"      > $INPUT    # repetition in x,y,z                                                           
      echo "1000"  >> $INPUT    # MC steps                                                                      
      echo "0.0"   >> $INPUT    # H field                                                                       
      echo "5"    >> $INPUT    # temperature                                                                   
      echo "1.9785"  >> $INPUT # local effective magnetic moment of Fe                                          
      echo "-0.1572"  >> $INPUT # local magnetic moment of B                                                    
      ;;
                                          
                                                                  
                                                               
                                 
                                              
    
# otherwise default
 *)
     echo "ID not in list, doing nothing"
esac
###########################











# wait a little bit, to get different seeds for the random numbers
T=`echo "$ID * 0.1 " | bc -l`
sleep $T
echo "sleeping for $T seconds"


# run program, if the input file exists
if [ -e "$INPUT" ] ; then ../../../heisenberg < $INPUT > $OUTFILE ; fi


# we are done!!!
echo "process $ID is done!"

