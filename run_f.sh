# ====================  office  ==========================
# ---------- train source model -------------
python train_src.py --trte val --output ckps/source/ --da oda --gpu_id 0 --dset office --max_epoch 100 --s 0
python train_src.py --trte val --output ckps/source/ --da oda --gpu_id 0 --dset office --max_epoch 100 --s 0
python train_src.py --trte val --output ckps/source/ --da oda --gpu_id 0 --dset office --max_epoch 100 --s 2

# ---------- train target model for adaptation -------------
python osda_train.py --da oda --dset office   --s 0 --t 1   --con_num_growth 0.1 --confi_nums 10  --reg_resid_para 1 --uncer_para 1 --reg_entro_para 0.1 
python osda_train.py --da oda --dset office   --s 0 --t 2   --con_num_growth 0.1 --confi_nums 10  --reg_resid_para 1 --uncer_para 1 --reg_entro_para 1 
python osda_train.py --da oda --dset office  --s 1 --t 0  --con_num_growth 0.2 --confi_nums 10  --reg_resid_para 1 --uncer_para 2      --reg_entro_para 0.1 
python osda_train.py --da oda --dset office  --s 1 --t 2   --con_num_growth 0.2 --confi_nums 10  --reg_resid_para 1 --uncer_para 1 --reg_entro_para 1  
python osda_train.py --da oda --dset office  --s 2 --t 0  --con_num_growth 0.2 --confi_nums 10  --reg_resid_para 1 --uncer_para 2      --reg_entro_para 0.1  
python osda_train.py --da oda --dset office   --s 2 --t 1   --con_num_growth 0.1 --confi_nums 10  --reg_resid_para 1 --uncer_para 1 --reg_entro_para 1  


# =====================  office-home  =====================
# ---------- train source model -------------
python train_src.py --trte val --output ckps/source/ --da oda --gpu_id 0 --dset office-home --max_epoch 50 --s 0
python train_src.py --trte val --output ckps/source/ --da oda --gpu_id 0 --dset office-home --max_epoch 50 --s 1
python train_src.py --trte val --output ckps/source/ --da oda --gpu_id 0 --dset office-home --max_epoch 50 --s 2
python train_src.py --trte val --output ckps/source/ --da oda --gpu_id 0 --dset office-home --max_epoch 50 --s 3

# ---------- train target model for adaptation -------------
python osda_train_ecml.py --da oda --dset office-home   --s 0 --t 1   --con_num_growth 0.1 --confi_nums 20  --reg_resid_para 1 --uncer_para 1 --reg_entro_para 0.1 --sigma 0.6   
python osda_train_ecml.py --da oda --dset office-home   --s 0 --t 2   --con_num_growth 0.1 --confi_nums 20  --reg_resid_para 1 --uncer_para 2 --reg_entro_para 0.1 --sigma 0.6 
python osda_train_ecml.py --da oda --dset office-home   --s 0 --t 3   --con_num_growth 0.1 --confi_nums 20  --reg_resid_para 1 --uncer_para 2 --reg_entro_para 0.1 --sigma 0.6  
python osda_train_ecml.py --da oda --dset office-home  --s 1 --t 0  --con_num_growth 0.1 --confi_nums 20  --reg_resid_para 1 --uncer_para 2      --reg_entro_para 0.1 --sigma 0.6  
python osda_train_ecml.py --da oda --dset office-home  --s 1 --t 2   --con_num_growth 0.1 --confi_nums 20  --reg_resid_para 1 --uncer_para 2 --reg_entro_para 0.1 --sigma 0.6  
python osda_train_ecml.py --da oda --dset office-home   --s 1 --t 3   --con_num_growth 0.1 --confi_nums 20  --reg_resid_para 1 --uncer_para 2 --reg_entro_para 0.1 --sigma 0.6 
python osda_train_ecml.py --da oda --dset office-home  --s 2 --t 0  --con_num_growth 0.1 --confi_nums 20  --reg_resid_para 1 --uncer_para 2      --reg_entro_para 0.1 --sigma 0.6  
python osda_train_ecml.py --da oda --dset office-home   --s 2 --t 1   --con_num_growth 0.1 --confi_nums 20  --reg_resid_para 1 --uncer_para 1 --reg_entro_para 0.1 --sigma 0.6  
python osda_train_ecml.py --da oda --dset office-home   --s 2 --t 3   --con_num_growth 0.1 --confi_nums 20  --reg_resid_para 1 --uncer_para 2 --reg_entro_para 0.1 --sigma 0.6  
python osda_train_ecml.py --da oda --dset office-home  --s 3 --t 0  --con_num_growth 0.1 --confi_nums 20  --reg_resid_para 1 --uncer_para 1      --reg_entro_para 0.1 --sigma 0.6  
python osda_train_ecml.py --da oda --dset office-home   --s 3 --t 1   --con_num_growth 0.1 --confi_nums 20  --reg_resid_para 1 --uncer_para 1 --reg_entro_para 0.1 --sigma 0.6  
python osda_train_ecml.py --da oda --dset office-home   --s 3 --t 2   --con_num_growth 0.1 --confi_nums 20  --reg_resid_para 1 --uncer_para 1 --reg_entro_para 0.1 --sigma 0.6  

 
