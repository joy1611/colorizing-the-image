parser.add_argument('--save_path', type=str, 
 
 default='./output_imgs', help='Save path for the test imgs')
   
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()
    return args