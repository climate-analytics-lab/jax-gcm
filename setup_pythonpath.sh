
root_dir=`pwd`
jcm_lib=$root_dir


echo "jcm lib path is: $jcm_lib"

if [ -d "$jcm_lib" ] ; then

    echo "Adding into PYTHONPATH"
    export PYTHONPATH=$jcm_lib:$PYTHONPATH
    echo "Updated PYTHONPATH=$PYTHONPATH"
else

    echo "Error: directory $jcm_lib does not exist"

fi

