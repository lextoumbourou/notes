# include <stdio.h>
# include <string.h>

static int const ROW_SIZE = 7;
static int const COL_SIZE = 7;
static int const MAX_SEQUENCE = 2;

static char const PLAYER_CHAR[2] = {'@', '%'};

void init_board(char board[ROW_SIZE][COL_SIZE]) {
	for (int i = 0; i <= ROW_SIZE; i++) {
		for (int j = 0; j <= COL_SIZE; j++) {
			board[i][j] = '0';
		}
	}
}

void show_title() {
	printf("\n=== Welcome to Connect4! ===\n");
}

void draw_board(char board[ROW_SIZE][COL_SIZE]) {
	printf("=== Board state! ===\n\n");

	printf("   |");
	for (int i = 0; i < COL_SIZE; i++) {
		printf(" %d |", i);
	}
	printf("\n");

	printf("    ");
	for (int i = 0; i < COL_SIZE; i++) {
		printf(" _  ");
	}
	printf("\n");


	for (int i = 0; i < ROW_SIZE; i++) {
		printf(" %d |", i);
		for (int j = 0; j < COL_SIZE; j++) {
			printf(" %c |", board[i][j]);
		}

		printf("\n");
	}

	printf("    ");
	for (int i = 0; i < COL_SIZE; i++) {
		printf(" _  ");
	}
	printf("\n");
}

int check_horizontal(char board[ROW_SIZE][COL_SIZE]) {
	int player_counters[] = {};
	int player_total = 0;
	int player;

	for (int i = 0; i < sizeof(PLAYER_CHAR); i++) {
		player_total += 1;
		player_counters[i] = 0;
	}

	for (int i = 0; i < ROW_SIZE; i++) {
		for (int j = 0; j < COL_SIZE; j++) {
			player = 0;

			for (player = 0; player < player_total; player++) {

				if (board[i][j] == PLAYER_CHAR[player]) {
					player_counters[player] += 1;
				} else {
					player_counters[player] = 0;
				}
			}

			if (player_counters[player] >= MAX_SEQUENCE) {
				return player;
			}
		}
	}

	return 0;
}

int check_board(char board[ROW_SIZE][COL_SIZE]) {
	// Check if 4 in a row horizontal
	int result;

	result = check_horizontal(board);
	if (result != 0) {
		return result;
	}

	// to do

	//result = check_vertical(board);
	//if (result != 0) {
	//	return result;
	//}

	//result = check_diagonal(board);
	//if (result != 0) {
	//	return result;
	//}


	return 0;
}

void take_turn(char board[ROW_SIZE][COL_SIZE], int player_num, char player_char) {
	int x, y;

	printf("Player %d's turn\n", player_num);

	printf("x-coordinate? \n");
	scanf("%d", &x);

	printf("y-coordinate? \n");
	scanf("%d", &y);

	board[y][x] = player_char;
}

int main() {
	char board[ROW_SIZE][COL_SIZE];
	int result, player;

	init_board(board);

	show_title();

	draw_board(board);

	while (1) {
		for (player = 0; player <= 2; player++) {
			take_turn(board, player, PLAYER_CHAR[player]);
			draw_board(board);
			result = check_board(board);
			if (result == player) {
				printf("Player %d wins!\n\n", player);
				return 0;
			}
		}
	}

	return 0;
}
