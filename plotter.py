
def plot_hml(x, title):
    x = np.array(x)
    plt.plot(x[:,1], label='median')
    plt.plot(x[:,0], c='lightblue', label='min')
    plt.plot(x[:,2], c='lightblue', label='max')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# training loss
plt.semilogy(train_losses)
plt.title('Training Loss')
plt.grid()
plt.show()

plot_hml(wmape_3hist, 'wMAPE')
plot_hml(smape_3hist, 'sMAPE')
plot_hml(cpears_3hist, 'Pearson Correlation')
plot_hml(cspear_3hist, 'Spearman Correlation')



def covar(x,y):
    return ((x-x.mean())*(y-y.mean())).mean()


# scatter and best-fit line
#       calib formula: prediction * slope + bias
#

m, b = np.polyfit(outputs.reshape(-1), targets.reshape(-1), 1)

plt.figure(figsize=(5,5))
plt.plot(np.linspace(0, outputs.max(), 11), m*np.linspace(0, outputs.max(), 11) + b, color = 'red')
plt.scatter(outputs.reshape(-1), targets.reshape(-1), s= 1)
plt.title('Scatterplot of Real and Predicted GMS')
plt.xlabel('predicted gms')
plt.ylabel('real gms')
plt.show()
print('Covarinace (no calib): {}'.format(covar(outputs.reshape(-1), targets.reshape(-1))))
print('Covarinace (calib all): {}'.format(covar(b+m*outputs.reshape(-1), targets.reshape(-1))))


corr_pears_list = []
corr_spear_list = []
smape_list = []
wmape_list = []
outputs = outputs.reshape(-1, FORECAST_SIZE)
targets = targets.reshape(-1, FORECAST_SIZE)
for i in range(FORECAST_SIZE):
    out, y = outputs[:,i], targets[:,i]
    corr_pears = np.corrcoef(out, y)[0,1]
    corr_spear, p = spearmanr(out, y)
    sMAPE = (abs(out-y)/((abs(out)+abs(y))))
    sMAPE = sMAPE[(1-np.isnan(sMAPE)).astype(bool)].mean()
    wMAPE = wmape(out, y)


    # outcalib = []
    # ycalib = []
    # for idx, outout, yy in zip((torch.cat(testX)[:,0,0].reshape(VOCAB_SIZE, -1).long()[:,0]).numpy(), out.reshape(VOCAB_SIZE, -1), y.reshape(VOCAB_SIZE, -1)):
    #     ls, lb = calibration_dict_item[idx]
    #     outcalib.append(outout*ls + lb)
    #     ycalib.append(yy)
    # plt.scatter(np.concatenate(outcalib), np.concatenate(ycalib), s=1)
    # plt.show()
    # print(np.corrcoef(np.concatenate(outcalib), np.concatenate(ycalib))[0,1])
    # print(spearmanr(np.concatenate(outcalib), np.concatenate(ycalib)))

    smape_list.append(sMAPE)
    wmape_list.append(wMAPE)
    corr_pears_list.append(corr_pears)
    corr_spear_list.append(corr_spear)


    # line_slope, line_bias = np.polyfit(outputs[:,i], targets[:,i], 1)
    # out_adj = (outputs[:,i] * line_slope) + line_bias
    # out_adj = np.maximum(out_adj, 0)
    # plt.scatter(out_adj, targets[:,i])

    plt.scatter(out, y, s= 1)
    plt.xlabel('pred_gms'); plt.ylabel('real_gms')
    plt.title(i)
    plt.grid()
    plt.show()
    print('sMAPE: {:<.2f}, wMAPE: {:<.2f}, CP: {:<.2f}, CS: {:<.2f}'.format(sMAPE*100, wMAPE*100, corr_pears, corr_spear))


plt.scatter(np.arange(FORECAST_SIZE),corr_pears_list)
plt.xticks(np.arange(FORECAST_SIZE))
plt.xlabel('Time steps')
plt.ylabel('Correlation(Y, Y_pred)')
plt.title('Pearson Correlation across future weeks')
plt.grid()
plt.show()

plt.scatter(np.arange(FORECAST_SIZE),corr_spear_list)
plt.xticks(np.arange(FORECAST_SIZE))
plt.xlabel('Time steps')
plt.ylabel('Correlation(Y, Y_pred)')
plt.title('Spearman Correlation across future weeks')
plt.grid()
plt.show()

plt.scatter(np.arange(FORECAST_SIZE),smape_list)
plt.xticks(np.arange(FORECAST_SIZE))
plt.xlabel('Time steps')
plt.ylabel('sMAPE')
plt.title('sMAPE across time')
plt.grid()
plt.show()
# individual scaling 0.58~0.7

plt.scatter(np.arange(FORECAST_SIZE),wmape_list)
plt.xticks(np.arange(FORECAST_SIZE))
plt.xlabel('Time steps')
plt.ylabel('wMAPE')
plt.title('wMAPE across time')
plt.grid()
plt.show()
