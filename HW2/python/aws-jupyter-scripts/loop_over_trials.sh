python Start_ec2_instance.py  
aws-jupyter run -s startup.sh
aws-jupyter send-dir --local ~/Vault/ --remote /home/ubuntu/Vault
# aws-jupyter send-dir --local ~/VaultDSC291/ --remote /home/ubuntu/Vault

# runtime="1440 minute"
# runtime="300 minute"
runtime="2 minute"
endtime=$(date -ud "$runtime" +%s)

while [[ $(date -u +%s) -le $endtime ]]
do
    echo "Time Now: `date +%H:%M:%S`"
    aws-jupyter run --script run_trial.py
    aws-jupyter retrieve --remote /home/ubuntu/workspace/results.pkl  --local from_remote/
    sleep 1m
    # sleep 60m
done