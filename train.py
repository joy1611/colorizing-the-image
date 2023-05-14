 sample_real_img_lab = real_img_lab
                    sample_fake_img_lab = fake_img_lab

            # display losses    
            # display losses
            print_losses(epoch_gen_adv_loss, epoch_gen_l1_loss,
                         epoch_disc_real_loss, epoch_disc_fake_loss,
                         epoch_disc_real_acc, epoch_disc_fake_acc,