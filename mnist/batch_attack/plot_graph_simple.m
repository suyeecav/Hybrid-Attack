function plot_graph_simple(query_budget,greedy_adv_num,rnd_adv_num,opt_adv_num)
%plot the greedy_search_performance
h = figure;
plot(0:query_budget,opt_adv_num(1:query_budget+1),0:query_budget,greedy_adv_num(1:query_budget+1),...
    0:query_budget,rnd_adv_num(1:query_budget+1),...
    'LineWidth',2);
grid on;
xlim([0  query_budget-1]);
ylim([0 Inf]);
%legend({'Greedy Search-Target Prob','Greedy Search-Loss Value','Random Search','Retroactive Optimal'},...
 %   'FontSize', 12,'Location','northoutside','Orientation','horizontal');
 legend({'Retroactive Optimal Strategy','Greedy Approach (Hybrid)','Random (Hybrid)', 'FontSize', 12);
xlabel('Budgets on Number of Queries','FontSize', 12);
ylabel('Number of Adversarial Samples Found','FontSize', 12);
