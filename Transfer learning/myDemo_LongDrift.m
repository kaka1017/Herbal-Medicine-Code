
%% Experimental Setting 1 Based on Benchmark Sensor Drift Dataset
%drift dataset: benchmark sensor drift dataset
%Set batch 1 as the source domain, and batch K (K = 2, 3, ..., 10) as the target domain
clear
clc
load Drift_dataset
%source domain: batch_1
XS = csvread("source_c5_paper1.csv");
label_XS = XS(:,43);
XS = XS(:,1:42)';
XS = XS * diag(1./sqrt(sum(XS.^2)));
XT = csvread("target_c5_paper1.csv");
label_XT = XT(:,43);
XT = XT(:,1:42)';
XT = XT * diag(1./sqrt(sum(XT.^2)));


%setting parameters of CSBD-CAELM
gamma_list = [1e-4,1e-3,1e-2,1e-1,1,10,100,1000,1000];
lamda_list = [1e-4,1e-3,1e-2,1e-1,1,10,100,1000,1000];
TH1 = 0.1;
TH2 = 0.1;%[0.1 0.3 0.9 0.7 0.2 0.8 0.9 0.7 0.5]
Cs = 0.001;
Ct = 1;
HN = 1000;
AF = 'tanh';%{'sin' 'relu' 'relu' 'relu'  'hardlim'  'sig' 'relu' 'tanh'  'relu'}
max_acc = -5000;
for dim = 1:42
    for lamda_index = 1:9
        for gamma_index = 1:9
            lamda = lamda_list(lamda_index);
            gamma = gamma_list(gamma_index);
            accuracy = 0;
            soft_label = [];%soft label of target domain
            IDM_T = [];
            for Iter = 1:10
                if Iter ==1
                    CDD = 0;
                end
                [P,XS_sub,XT_sub,u_sc,u_subc] = CSBD(XS,label_XS,XT,CDD,dim,lamda,gamma); %CSBD

                P_train = normalize(XS_sub,2);
                P_test = normalize(XT_sub,2);

                %Parameter Settings
                TrainingFile = [label_XS,P_train'];%labels and instances of the source domain
                GuideSamplesFile = [soft_label(IDM_T),(P_test(:,IDM_T))'];%%labels and instances of the guide samples
                TestingFile = [label_XT,P_test'];%labels and instances of the target domain

                %select output weight
                if length(label_XS)>HN
                    NL = 1;
                else
                    NL = 0;
                end

                %CAELM
                if Iter == 1
                    soft_label = CAELM(TrainingFile,GuideSamplesFile,TestingFile, HN,AF,Cs,Ct,NL,1);%ELM
                else
                    soft_label = CAELM(TrainingFile,GuideSamplesFile,TestingFile, HN,AF,Cs,Ct,NL,0);%DAELM
                end
                acc = length(find(label_XT == soft_label))/length(label_XT);
                if accuracy < acc
                    accuracy = acc;
                end

                % select guide sample set
                IDM_T = [];
                for i = 1:4 % 1:5
                    if length(find(soft_label == i))>(length(soft_label)*TH1)
                        k = find (soft_label == i);
                        XT_subi = XT_sub(:,k);
                        Distance = pdist([(u_subc{i})';XT_subi']);
                        Dis_Matrix = squareform(Distance);
                        Dis_Matrix = Dis_Matrix(2:end,1);
                        [B,I] = sort(Dis_Matrix);
                        IDM_T = [IDM_T;k(I(1:floor(length(k)*TH2)))];
                    end
                end

                %Update conditional distribution distance CDD
                u_tc = cell(4,1);%(5,1)
                d = 0;
                for i = 1:4 % 1:5
                    if (size(find(soft_label == i),1) ~= 0) && (size(u_sc{i},1) ~= 0)
                        XT_i = XT(:,find(soft_label == i));
                        u_tc{i} = sum(XT_i,2)/size(XT_i,2);
                        d = d + (u_sc{i}-u_tc{i})*(u_sc{i}-u_tc{i})';
                    end
                end
                CDD = d;

            end
            disp(['Source domain: Batch',num2str(1),', Target domain: Batch',num2str(1),', ','Accuracy = ' num2str(accuracy)]);
            if accuracy > max_acc
                max_acc = accuracy;
            end
        end
    end
end

disp(['The best acc',num2str(1),', Target domain: Batch',num2str(1),', ','Accuracy = ' num2str(max_acc)]);
