

clear all;
close all;
%
addpath('Utilities');
addpath('Dataset');
%
load('.\Dataset\COIL20_32x32.mat')
fea = double(fea); nnClass =  length(unique(gnd));
num_Class = []; sele_num = 15;
for i = 1:nnClass
    num_Class = [num_Class length(find(gnd==i))];
end
Tr_DAT  = []; trls = [];
Tt_DAT   = []; ttls  = [];
for j = 1:nnClass
    idx = find(gnd==j);
    Tr_DAT = [Tr_DAT;fea(idx((1:sele_num)),:)];
    trls= [trls;gnd(idx((1:sele_num)))];
    Tt_DAT  = [Tt_DAT;fea(idx((sele_num+1:num_Class(j))),:)];
    ttls = [ttls;gnd(idx((sele_num+1:num_Class(j))))];
end
Tr_DAT = Tr_DAT'/256; trls = trls';
Tt_DAT = Tt_DAT'/256; ttls = ttls';
Image_row_NUM = 32; Image_column_NUM = 32;
%

opts.row = Image_row_NUM; opts.col = Image_column_NUM;
tr_dat  =  Tr_DAT;
tt_dat  =  Tt_DAT;
% normalization
tr_descr  =  tr_dat./( repmat(sqrt(sum(tr_dat.*tr_dat)), [size(tr_dat,1),1]) );
tt_descr  =  tt_dat./( repmat(sqrt(sum(tt_dat.*tt_dat)), [size(tr_dat,1),1]) );
Train_Ma = tr_descr; Train_Lab = trls';
Test_Ma  = tt_descr; Test_Lab = ttls';

%%
ImageAcc = zeros(6, 1);
ImageRate = zeros(6, 6);
Promethods = {'S1DLRR', 'S23DLRR',  'S12DLRR'};
for fun_num = 1% : length(Promethods)
    mymethod = Promethods{fun_num};
    disp([' choosemethod = '  mymethod]);
    %
    %mu_num =   [3 5 10 15 20 25 30];
    %for kkmu = 1 %:  %length(mu_num)
    opts.mu = 3;%mu_num(kkmu);
    %disp([ ' mu = ' num2str(opts.mu)]);

    rank_num=  [3 5 8 10 13 15];
    lam_num =  [0.0001 0.001 0.01 0.1 0.5 1.0];
    %
    for kkrank = 1 : length(rank_num)
        opts.rank = rank_num(kkrank);
        disp([ ' rank = ' num2str(opts.rank)]);
        % lambda
        for kk = 1 : length(lam_num)
            opts.lambda = lam_num(kk);

            opts.nnClass = nnClass;
            opts.sele_num = sele_num;

            %
            tic;
            switch (mymethod)
                case 'S1DLRR'
                    [X, E, iter] = S1DLRR(Train_Ma, Test_Ma, opts);
                case 'S23DLRR'
                    [X, E, iter] = S23DLRR(Train_Ma, Test_Ma, opts);
                case 'S12DLRR'
                    [X, E, iter] = S12DLRR(Train_Ma, Test_Ma, opts);
            end
            time_cost = toc;
            %
            [D, class_test] = spnf_classify(Train_Ma, Test_Ma, X, opts, fun_num);
            acc_test = sum(Test_Lab == class_test')/length(Test_Lab)*100;
            %
            disp([ ' lambda = ' num2str(opts.lambda), ' acc_result= ' num2str(acc_test), ' time_cost = ' num2str(time_cost)]);

            ImageAcc(kk) = acc_test;

        end

        ImageRate(:,kkrank)  =  ImageAcc;
    end

    eval(['ImageRateCOIL20_' mymethod '= ImageRate']);
    eval(['save ImageRateCOIL20_' mymethod ' ImageRateCOIL20_' mymethod]);

end
%end

