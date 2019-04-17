import CnnModel


class Trainer:
    def __init__(self, model_info):
        self.cnn = CnnModel.CnnModel(model_info)

    def train(self):
        image_paths, labels = self.cnn.get_image_paths_and_labels_from_image_files()
        self.cnn.create_batch(image_paths, labels)
        self.cnn.create_model()
        print("start training...")
        epoch_accuracy = 0
        epoch_step = 0
        for step in range(1, 100000):
            _, loss, accuracy = self.cnn.sess.run([self.cnn.minimizer, self.cnn.loss, self.cnn.accuracy])
            epoch_accuracy += float(accuracy)

            if step % 5 == 0:
                print("step:" + str(step), epoch_accuracy / epoch_step, loss)

            if step % 100 == 0:
                self.cnn.saver.save(self.cnn.sess, self.cnn.model_path + r'\model.ckpt', global_step=step)
                epoch_accuracy = 0
                epoch_step = 0
            epoch_step += 1


if __name__ == "__main__":
    trainer = Trainer(CnnModel.MODEL_INFO)
    trainer.train()
