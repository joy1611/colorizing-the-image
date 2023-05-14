def print_losses(epoch_gen_adv_loss, epoch_gen_l1_loss, epoch_disc_real_loss, epoch_disc_fake_loss,
                 epoch_disc_real_acc, epoch_disc_fake_acc, data_loader_len, l1_weight): 	                 epoch_disc_real_acc, epoch_disc_fake_acc, data_loader_len, l1_weight):

    print('  Generator: adversarial loss = {:.4f}, L1 loss = {:.4f}, full loss = {:.4f}'.format(	    print('  Generator: adversarial loss = {:.4f}, L1 loss = {:.4f}, full loss = {:.4f}'.format(
        epoch_gen_adv_loss / data_loader_len,	        epoch_gen_adv_loss / data_loader_len,
@@ -31,10 +31,10 @@ def save_sample(real_imgs_lab, fake_imgs_lab, save_path, plot_size=20, scale=2.2


    # create white canvas	    # create white canvas
    canvas = np.ones((3*32 + 4*6, plot_size*32 + (plot_size+1)*6, 3), dtype=np.uint8)*255	    canvas = np.ones((3*32 + 4*6, plot_size*32 + (plot_size+1)*6, 3), dtype=np.uint8)*255
    	
    real_imgs_lab = real_imgs_lab.cpu().numpy()	    real_imgs_lab = real_imgs_lab.cpu().numpy()
    fake_imgs_lab = fake_imgs_lab.cpu().numpy()	    fake_imgs_lab = fake_imgs_lab.cpu().numpy()
    	
    for i in range(0, plot_size):	    for i in range(0, plot_size):
        # postprocess real and fake samples	        # postprocess real and fake samples
        real_bgr = postprocess(real_imgs_lab[i])	        real_bgr = postprocess(real_imgs_lab[i])
@@ -46,11 +46,11 @@ def save_sample(real_imgs_lab, fake_imgs_lab, save_path, plot_size=20, scale=2.2
        canvas[44:76, x:x+32, :] = np.repeat(grayscale, 3, axis=2)	        canvas[44:76, x:x+32, :] = np.repeat(grayscale, 3, axis=2)
        canvas[82:114, x:x+32, :] = fake_bgr	        canvas[82:114, x:x+32, :] = fake_bgr


    # scale 	    # scale
    canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)	    canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    # save 	    # save
    cv2.imwrite(os.path.join(save_path), canvas)	    cv2.imwrite(os.path.join(save_path), canvas)
    	
    if show:	    if show:
        cv2.destroyAllWindows()	        cv2.destroyAllWindows()
        cv2.imshow('sample', canvas)	        cv2.imshow('sample', canvas)
@@ -67,16 +67,16 @@ def save_test_sample(real_imgs_lab, fake_imgs_lab1, fake_imgs_lab2, save_path, p


    # create white canvas	    # create white canvas
    canvas = np.ones((plot_size*32 + (plot_size+1)*6, 4*32 + 5*8, 3), dtype=np.uint8)*255	    canvas = np.ones((plot_size*32 + (plot_size+1)*6, 4*32 + 5*8, 3), dtype=np.uint8)*255
    	
    real_imgs_lab = real_imgs_lab.cpu().numpy()	    real_imgs_lab = real_imgs_lab.cpu().numpy()
    fake_imgs_lab1 = fake_imgs_lab1.cpu().numpy()	    fake_imgs_lab1 = fake_imgs_lab1.cpu().numpy()
    fake_imgs_lab2 = fake_imgs_lab2.cpu().numpy()	    fake_imgs_lab2 = fake_imgs_lab2.cpu().numpy()
    	
    for i in range(0, plot_size):	    for i in range(0, plot_size):
        # post-process real and fake samples	        # post-process real and fake samples
        real_bgr = postprocess(real_imgs_lab[i])	        real_bgr = postprocess(real_imgs_lab[i])
        fake_bgr1 = postprocess(fake_imgs_lab1[i])  	        fake_bgr1 = postprocess(fake_imgs_lab1[i])
        fake_bgr2 = postprocess(fake_imgs_lab2[i])     	        fake_bgr2 = postprocess(fake_imgs_lab2[i])
        grayscale = np.expand_dims(cv2.cvtColor(real_bgr.astype(np.float32), cv2.COLOR_BGR2GRAY), 2)	        grayscale = np.expand_dims(cv2.cvtColor(real_bgr.astype(np.float32), cv2.COLOR_BGR2GRAY), 2)
        # paint	        # paint
        x = (i+1)*6+i*32	        x = (i+1)*6+i*32
@@ -85,11 +85,11 @@ def save_test_sample(real_imgs_lab, fake_imgs_lab1, fake_imgs_lab2, save_path, p
        canvas[x:x+32, 88:120, :] = fake_bgr1	        canvas[x:x+32, 88:120, :] = fake_bgr1
        canvas[x:x+32, 128:160, :] = fake_bgr2	        canvas[x:x+32, 128:160, :] = fake_bgr2


    # scale 	    # scale
    canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)	    canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    # save 	    # save
    cv2.imwrite(os.path.join(save_path), canvas)	    cv2.imwrite(os.path.join(save_path), canvas)
    	
    if show:	    if show:
        cv2.destroyAllWindows()	        cv2.destroyAllWindows()
        cv2.imshow('sample', canvas)	        cv2.imshow('sample', canvas)
@@ -109,6 +109,6 @@ def adjust_learning_rate(optimizer, global_step, base_lr, lr_decay_rate=0.1, lr_
    lr = base_lr * (lr_decay_rate ** (global_step/lr_decay_steps))	    lr = base_lr * (lr_decay_rate ** (global_step/lr_decay_steps))
    if lr < 1e-6:	    if lr < 1e-6:
        lr = 1e-6	        lr = 1e-6
    	
    for param_group in optimizer.param_groups:	    for param_group in optimizer.param_groups:
        param_group['lr'] = lr	        param_group['lr'] = lr
  36 changes: 21 additions & 15 deletions36  
n