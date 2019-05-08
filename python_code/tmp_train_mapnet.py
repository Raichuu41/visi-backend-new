def train_mapnet(model, features, labels, weights=None,
                 test_features=None, test_labels=None,
                 lr=1e-3, batch_size=2000, batch_frac_labeled=0.7,
                 random_state=123, use_gpu=True,
                 verbose=False, outpath=None, log_dir=None, max_epochs=float('inf'),
                 plot_fn=None):
    use_test = test_features is not None and test_labels is not None
    use_multi_features = features.ndim == 3      # multi features have shape FTS_VARIANTS X N_SAMPLES X FEATURE_DIM
    multi_features = None if not use_multi_features else features.copy()

    def sample_features():
        features = multi_features.copy()

        # pick a random feature variant for each sample
        features = np.stack([
            features[np.random.randint(0, len(features)), i, :]
            for i in range(features.shape[1])
        ])

        return features

    if use_multi_features:
        features = sample_features()

    if not isinstance(labels, torch.LongTensor):
        labels = torch.LongTensor(labels)
    if use_test and not isinstance(test_labels, torch.LongTensor):
        test_labels = torch.LongTensor(test_labels)
    if weights is not None and not isinstance(weights, torch.FloatTensor):
        weights = torch.FloatTensor(weights)

    def eval_fn(input):
        fts = model.mapping.forward(input)
        fts = fts / fts.norm(dim=1, keepdim=True)
        proj = model.embedder.forward(fts)

        return fts, proj

    if outpath is not None and not os.path.isdir(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath))

    if log_dir is not None:
        logger = TBPlotter(log_dir)
        logger.print_logdir()
    else:
        logger = None

    loader_kwargs = {'num_workers': 4} if use_gpu else {}

    N_unlabeled_total = batch_size
    classweights = {l: 1./3 if l < 0 else 2./3 for l in set(labels.numpy()) if l != 0}      # 2/3 positive labeled, 1/3 class specific negatives
    batchsampler = PartiallyLabeledBatchSampler(labels, frac_labeled=batch_frac_labeled, batch_size=batch_size,
                                                N_unlabeled_total=N_unlabeled_total, classweights=classweights)

    trainloader = DataLoader(IndexDataset(features), batch_sampler=batchsampler,
                             **loader_kwargs)
    if use_test:
        testloader = DataLoader(IndexDataset(test_features), batch_size=batch_size, shuffle=False, drop_last=False,
                                **loader_kwargs)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=2e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, threshold=1e-3,
                                                        verbose=verbose)

    triplet_selector = TripletSelector(margin=torch.tensor(0.2), negative_selection_fn=select_semihard)
    loss_fns_train = [
        TripletLossWrapper(triplet_selector, labels, data_weights=weights,
                           N_labels_per_minibatch=10, N_samples_per_label=5, N_pure_negatives=15,
                           average=True),
        TSNEWrapperMapNet(N=len(features), use_gpu=use_gpu),
        # L1RegWrapper(model=model.mapping)
    ]
    if use_test:
        triplet_selector_test = TripletSelector(margin=torch.tensor(0.2), negative_selection_fn=select_random)
        loss_fns_test = [
            TripletLossWrapper(triplet_selector_test, test_labels,
                               N_labels_per_minibatch=10, N_samples_per_label=5, N_pure_negatives=15,
                               average=True),
            TSNEWrapperMapNet(N=len(test_features), use_gpu=use_gpu)
        ]

    # train network until scheduler reduces learning rate to threshold value
    lr_threshold = 5e-5
    epoch = 1

    best_loss = float('inf')
    best_loss_test = float('inf')
    best_epoch = -1
    stopcriterion = False
    while not stopcriterion:
        if use_multi_features:          # for each epoch sample random feature variants
            features = sample_features()

        # compute beta for mapped features
        if verbose:
            print('Compute beta for TSNE-Loss...')
        mapped_features = torch.from_numpy(features) if epoch == 1 else \
            evaluate_model(model.mapping, features, batch_size=batch_size, use_gpu=use_gpu, verbose=verbose)
        mapped_features /= mapped_features.norm(dim=1, keepdim=True)
        loss_fns_train[1]._compute_beta(mapped_features)
        if verbose:
            print('Done.')
        model.train()
        trainlosses = train(eval_fn, trainloader, optimizer, loss_fns_train, epoch, loss_fns_weights=(100, 1),
                            use_gpu=use_gpu, verbose=verbose, logger=logger)
        trainloss = trainlosses[-1]
        if use_test:
            model.eval()
            testlosses = test(eval_fn, testloader, loss_fns_test, epoch, loss_fns_weights=(100, 1),
                              use_gpu=use_gpu, verbose=verbose, logger=logger)
            testloss = testlosses[-1]
        else:
            lr_scheduler.step(trainloss)
        if trainloss < best_loss:
            best_loss = trainloss
            best_epoch = epoch
            if outpath is not None:
                torch.save({                # save best model
                    'state_dict': model.state_dict(),
                }, outpath)

        if use_test and testloss < best_loss_test:
            best_loss_test = testloss
            if outpath is not None:
                torch.save({                # save best model
                    'state_dict': model.state_dict(),
                }, outpath.replace('.pth.tar', '_test.pth.tar'))

        if log_dir is not None:         # save checkpoint
            torch.save({
                'epoch': epoch,
                'best_loss': best_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict()
            }, os.path.join(log_dir, 'checkpoint.pth.tar'))

        if plot_fn is not None:         # compute 2-dimensional embedding and plot it with plot_fn
            projection = evaluate_model(model.forward, data=features, batch_size=batch_size, use_gpu=use_gpu, verbose=False)
            plot_fn(projection.numpy(), epoch=epoch)

        epoch += 1
        if epoch > max_epochs:
            break
        stopcriterion = (optimizer.param_groups[-1]['lr'] < lr_threshold) or \
                        (trainlosses[0] < 1)

    # refine projection
    best_weights = load_weights(outpath, model.state_dict())
    model.load_state_dict(best_weights)
    model.eval()
    mapped_features = evaluate_model(model.mapping, features, batch_size=batch_size, use_gpu=use_gpu, verbose=verbose)
    mapped_features /= mapped_features.norm(dim=1, keepdim=True)
    train_embedder(model.embedder, mapped_features.cpu().numpy(), lr=1e-3, batch_size=2000,
                   random_state=random_state, use_gpu=use_gpu,
                   verbose=verbose, outpath=None, log_dir=None, plot_fn=plot_fn)

    # save model with refined projection
    if outpath is not None:
        torch.save({
            'state_dict': model.state_dict(),
        }, outpath)

    print('Finished training mapnet with best training loss: {:.4f} from epoch {}.'.format(best_loss, best_epoch))
    if outpath is not None:
        print('Saved model to {}.'.format(outpath))